from flask import Flask, request, jsonify, send_file
import json
import uuid
import os
import threading
import time
import requests
import ffmpeg
import pysrt
from datetime import datetime
from urllib.parse import urlparse
import tempfile
import re
import subprocess
import sys

# Импорты для создания прокручивающихся субтитров
try:
    import whisper
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    print("Warning: Whisper не установлен. Автоматическое создание субтитров недоступно.")

app = Flask(__name__)

# Конфигурация
TEMP_DIR = "temp_files"
OUTPUT_DIR = "output_videos"
TASKS = {}

# Создание директорий
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

class VideoDownloader:
    """Класс для загрузки видео с xAPI заголовками"""
    
    def __init__(self):
        self.session = requests.Session()
        # Установка базовых заголовков для xAPI
        self.session.headers.update({
            'User-Agent': 'xAPI-Video-Processor/1.0',
            'X-Experience-API-Version': '1.0.3',
            'Content-Type': 'application/json',
            'Accept': 'application/json, video/mp4, */*'
        })
    
    def download_video(self, video_url, output_path, xapi_headers=None):
        """Загрузка видео с xAPI заголовками"""
        try:
            headers = self.session.headers.copy()
            if xapi_headers:
                headers.update(xapi_headers)
            
            response = self.session.get(video_url, headers=headers, stream=True)
            response.raise_for_status()
            
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            return True
            
        except Exception as e:
            print(f"Ошибка загрузки видео {video_url}: {e}")
            return False

class VideoProcessor:
    def __init__(self, task_id, video_data, subtitles_content=None):
        self.task_id = task_id
        self.video_data = video_data
        self.subtitles_content = subtitles_content
        self.status = "processing"
        self.progress = 0
        self.downloader = VideoDownloader()
        
    def process_videos(self):
        """Основная функция обработки видео"""
        try:
            TASKS[self.task_id]["status"] = "processing"
            self.progress = 0
            TASKS[self.task_id]["progress"] = self.progress
            
            # Шаг 1: Загрузка и объединение видео частей (20%)
            combined_video_path = self.download_and_combine_videos()
            if not combined_video_path:
                TASKS[self.task_id]["status"] = "error"
                TASKS[self.task_id]["error"] = "Failed to download and combine videos."
                return
            self.progress = 20
            TASKS[self.task_id]["progress"] = self.progress

            # Шаг 2: Проверка наличия аудио
            has_audio = self.check_audio_exists(combined_video_path)
            
            # Шаг 3: Обработка субтитров (50%)
            word_level_info = None
            line_level_subtitles = None
            
            if has_audio and WHISPER_AVAILABLE:
                # Извлечение аудио и создание субтитров с временными метками слов
                word_level_info = self.extract_word_timestamps(combined_video_path)
                if word_level_info:
                    line_level_subtitles = self.convert_to_line_level(word_level_info)
                    print(f"Задача {self.task_id}: Создано {len(line_level_subtitles)} строк субтитров из аудио")
            
            # Если аудио нет или нет word_level_info, используем SRT из webhook
            if not word_level_info and self.subtitles_content:
                subtitles_path = self.save_subtitles_from_webhook(self.subtitles_content)
                line_level_subtitles = self.parse_srt_to_line_level(subtitles_path)
                print(f"Задача {self.task_id}: Используются субтитры из webhook")
            
            # Если ничего нет, создаем из описания
            if not line_level_subtitles:
                full_text = self.extract_text_from_videos()
                if full_text:
                    line_level_subtitles = self.generate_subtitles_from_text(full_text)
                    print(f"Задача {self.task_id}: Субтитры сгенерированы из описания")
            
            self.progress = 70
            TASKS[self.task_id]["progress"] = self.progress
            
            # Шаг 4: Создание финального видео с прокручивающимися субтитрами (30%)
            final_video_path = self.create_scrolling_subtitles_video(
                combined_video_path, 
                word_level_info, 
                line_level_subtitles,
                has_audio
            )
            
            if not final_video_path:
                TASKS[self.task_id]["status"] = "error"
                TASKS[self.task_id]["error"] = "Failed to create final video with scrolling subtitles."
                return
            
            self.progress = 100
            TASKS[self.task_id]["progress"] = self.progress
            
            # Завершение
            TASKS[self.task_id]["status"] = "completed"
            TASKS[self.task_id]["output_file"] = final_video_path
            print(f"Задача {self.task_id} успешно завершена. Файл: {final_video_path}")
            
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            TASKS[self.task_id]["status"] = "error"
            TASKS[self.task_id]["error"] = str(e)
            print(f"Критическая ошибка в задаче {self.task_id}: {e}")
            print(f"DEBUG [{self.task_id}]: Full error traceback: {error_trace}")

    def check_audio_exists(self, video_path):
        """Проверка наличия аудиодорожки в видео"""
        try:
            probe = ffmpeg.probe(video_path)
            return any(stream['codec_type'] == 'audio' for stream in probe.get('streams', []))
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка при проверке аудио: {e}")
            return False

    def extract_word_timestamps(self, video_path):
        """Извлечение временных меток слов с помощью Whisper"""
        try:
            if not WHISPER_AVAILABLE:
                return None
                
            # Извлечение аудио
            audio_path = os.path.join(TEMP_DIR, f"audio_{self.task_id}.wav")
            (ffmpeg
             .input(video_path)
             .audio
             .output(audio_path)
             .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
            
            # Использование Faster Whisper для получения временных меток слов
            model = WhisperModel("base", device="cpu")
            segments, info = model.transcribe(audio_path, word_timestamps=True)
            
            word_level_info = []
            for segment in segments:
                for word in segment.words:
                    word_level_info.append({
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end
                    })
            
            # Очистка временного аудио файла
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return word_level_info
            
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка извлечения временных меток: {e}")
            return None

    def convert_to_line_level(self, word_level_info, max_chars=47):
        """Конвертация временных меток слов в строки"""
        line_level_subtitles = []
        current_line = ""
        line_start = None
        line_end = None
        
        for word_info in word_level_info:
            word = word_info['word'].strip()
            
            if line_start is None:
                line_start = word_info['start']
            
            # Проверка длины строки
            if len(current_line + word) <= max_chars:
                current_line += word + " "
                line_end = word_info['end']
            else:
                # Сохранение текущей строки и начало новой
                if current_line.strip():
                    line_level_subtitles.append({
                        'text': current_line.strip(),
                        'start': line_start,
                        'end': line_end
                    })
                
                current_line = word + " "
                line_start = word_info['start']
                line_end = word_info['end']
        
        # Добавление последней строки
        if current_line.strip():
            line_level_subtitles.append({
                'text': current_line.strip(),
                'start': line_start,
                'end': line_end
            })
        
        return line_level_subtitles

    def parse_srt_to_line_level(self, srt_path):
        """Парсинг SRT файла в формат line_level"""
        try:
            line_level_subtitles = []
            subs = pysrt.open(srt_path)
            
            for sub in subs:
                start_seconds = (sub.start.hours * 3600 + 
                               sub.start.minutes * 60 + 
                               sub.start.seconds + 
                               sub.start.milliseconds / 1000.0)
                
                end_seconds = (sub.end.hours * 3600 + 
                             sub.end.minutes * 60 + 
                             sub.end.seconds + 
                             sub.end.milliseconds / 1000.0)
                
                line_level_subtitles.append({
                    'text': sub.text.replace('\n', ' '),
                    'start': start_seconds,
                    'end': end_seconds
                })
            
            return line_level_subtitles
            
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка парсинга SRT: {e}")
            return None

    def create_scrolling_subtitles_video(self, video_path, word_level_info, line_level_subtitles, has_audio):
        """Создание видео с прокручивающимися субтитрами"""
        final_output_path = os.path.join(OUTPUT_DIR, f"final_scrolling_subtitles_{self.task_id}.mp4")
        
        try:
            # Создание субтитров в формате ASS для прокручивания
            ass_path = self.create_ass_subtitles(word_level_info, line_level_subtitles)
            
            video_input = ffmpeg.input(video_path)
            
            if ass_path and os.path.exists(ass_path):
                # Применение ASS субтитров с прокручиванием
                video_with_subs = ffmpeg.filter(
                    video_input.video,
                    'subtitles',
                    ass_path
                )
                
                streams_to_output = [video_with_subs]
                if has_audio:
                    streams_to_output.append(video_input.audio)
                
                output_params = {
                    'vcodec': 'libx264',
                    'preset': 'medium',
                    'crf': 23,
                    'pix_fmt': 'yuv420p'
                }
                
                if has_audio:
                    output_params['acodec'] = 'aac'
                    output_params['audio_bitrate'] = '128k'
                
                (ffmpeg
                 .output(*streams_to_output, final_output_path, **output_params)
                 .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
            else:
                # Fallback без субтитров
                streams_to_output = [video_input.video]
                if has_audio:
                    streams_to_output.append(video_input.audio)
                
                copy_params = {'vcodec': 'copy'}
                if has_audio:
                    copy_params['acodec'] = 'copy'
                
                (ffmpeg
                 .output(*streams_to_output, final_output_path, **copy_params)
                 .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
            
            print(f"Задача {self.task_id}: Финальное видео с прокручивающимися субтитрами сохранено в {final_output_path}")
            return final_output_path
            
        except ffmpeg.Error as e:
            stderr_output = e.stderr.decode('utf8') if e.stderr else "No stderr"
            print(f"Задача {self.task_id}: Ошибка FFmpeg: {stderr_output}")
            return self.create_fallback_video(video_path, line_level_subtitles, has_audio)
        except Exception as e:
            print(f"Задача {self.task_id}: Общая ошибка создания видео: {e}")
            return self.create_fallback_video(video_path, line_level_subtitles, has_audio)

    def create_ass_subtitles(self, word_level_info, line_level_subtitles):
        """Создание ASS файла с анимированными субтитрами"""
        try:
            ass_path = os.path.join(TEMP_DIR, f"subtitles_{self.task_id}.ass")
            
            # ASS заголовок
            ass_content = """[Script Info]
Title: Scrolling Subtitles
ScriptType: v4.00+

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,32,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,3,2,30,30,50,1
Style: Highlight,Arial,32,&H0000FFFF,&H000000FF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,2,3,2,30,30,50,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
            
            if line_level_subtitles:
                for line in line_level_subtitles:
                    start_time = self.seconds_to_ass_time(line['start'])
                    end_time = self.seconds_to_ass_time(line['end'])
                    
                    # Добавление основного текста строки
                    ass_content += f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{line['text']}\n"
                    
                    # Добавление подсветки слов если есть word_level_info
                    if word_level_info:
                        self.add_word_highlights(ass_content, line, word_level_info)
            
            with open(ass_path, 'w', encoding='utf-8') as f:
                f.write(ass_content)
            
            return ass_path
            
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка создания ASS файла: {e}")
            return None

    def seconds_to_ass_time(self, seconds):
        """Конвертация секунд в формат времени ASS"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        return f"{hours:01d}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def add_word_highlights(self, ass_content, line, word_level_info):
        """Добавление подсветки отдельных слов"""
        words_in_line = line['text'].split()
        
        for word_info in word_level_info:
            word = word_info['word'].strip()
            
            # Проверка, принадлежит ли слово текущей строке
            if (line['start'] <= word_info['start'] <= line['end'] and 
                word.lower() in [w.lower() for w in words_in_line]):
                
                start_time = self.seconds_to_ass_time(word_info['start'])
                end_time = self.seconds_to_ass_time(word_info['end'])
                
                # Вычисление позиции слова для анимации
                word_index = next((i for i, w in enumerate(words_in_line) 
                                 if w.lower() == word.lower()), 0)
                
                # Создание эффекта прокручивания для слова
                highlight_text = self.create_word_highlight_effect(word, word_index, len(words_in_line))
                
                ass_content += f"Dialogue: 1,{start_time},{end_time},Highlight,,0,0,0,,{highlight_text}\n"

    def create_word_highlight_effect(self, word, word_index, total_words):
        """Создание эффекта подсветки для отдельного слова"""
        # Расчет позиции слова
        x_offset = (word_index - total_words / 2) * 40  # Приблизительное смещение
        
        # ASS тэги для анимации и позиционирования
        return f"{{\\pos(960{x_offset:+d},540)\\c&H0000FFFF&\\t(\\c&H00FFFFFF&)}}{word}"

    def create_fallback_video(self, video_path, line_level_subtitles, has_audio):
        """Fallback создание видео с простыми субтитрами"""
        try:
            final_output_path = os.path.join(OUTPUT_DIR, f"final_fallback_{self.task_id}.mp4")
            
            video_input = ffmpeg.input(video_path)
            
            if line_level_subtitles:
                # Создание простого SRT файла
                srt_path = self.create_simple_srt(line_level_subtitles)
                
                if srt_path:
                    processed_video_stream = ffmpeg.filter(
                        video_input.video, 
                        'subtitles', 
                        srt_path,
                        force_style='FontName=Arial,FontSize=28,PrimaryColour=&H00FFFFFF,BackColour=&H80000000,BorderStyle=1,Outline=2'
                    )
                    
                    streams_to_output = [processed_video_stream]
                    if has_audio:
                        streams_to_output.append(video_input.audio)
                    
                    output_params = {'vcodec': 'libx264', 'preset': 'fast', 'crf': 23}
                    if has_audio:
                        output_params['acodec'] = 'aac'
                    
                    (ffmpeg
                     .output(*streams_to_output, final_output_path, **output_params)
                     .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
                else:
                    # Копирование без субтитров
                    self.copy_video_without_subtitles(video_input, final_output_path, has_audio)
            else:
                # Копирование без субтитров
                self.copy_video_without_subtitles(video_input, final_output_path, has_audio)
            
            return final_output_path
            
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка в fallback методе: {e}")
            raise Exception(f"Критическая ошибка: невозможно создать видео: {e}")

    def copy_video_without_subtitles(self, video_input, output_path, has_audio):
        """Копирование видео без субтитров"""
        streams_to_output = [video_input.video]
        if has_audio:
            streams_to_output.append(video_input.audio)
        
        copy_params = {'vcodec': 'copy'}
        if has_audio:
            copy_params['acodec'] = 'copy'
        
        (ffmpeg
         .output(*streams_to_output, output_path, **copy_params)
         .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))

    def create_simple_srt(self, line_level_subtitles):
        """Создание простого SRT файла из line_level_subtitles"""
        try:
            srt_path = os.path.join(TEMP_DIR, f"simple_subtitles_{self.task_id}.srt")
            
            srt_content = ""
            for i, line in enumerate(line_level_subtitles, 1):
                start_time = self.format_srt_time(line['start'])
                end_time = self.format_srt_time(line['end'])
                
                srt_content += f"{i}\n"
                srt_content += f"{start_time} --> {end_time}\n"
                srt_content += f"{line['text']}\n\n"
            
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            
            return srt_path
            
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка создания SRT: {e}")
            return None

    def format_srt_time(self, seconds):
        """Форматирование времени для SRT"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millisecs = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"

    # Остальные методы без изменений...
    def save_subtitles_from_webhook(self, subtitles_content):
        """Сохранение субтитров, переданных через webhook"""
        try:
            srt_path = os.path.join(TEMP_DIR, f"webhook_subtitles_{self.task_id}.srt")
            with open(srt_path, 'w', encoding='utf-8') as f:
                f.write(subtitles_content)
            return srt_path
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка сохранения субтитров из webhook: {e}")
            return None

    def download_and_combine_videos(self):
        """Загрузка видео по URL и объединение"""
        downloaded_files = []
        
        for i, video_info in enumerate(self.video_data):
            video_url = self.extract_video_url(video_info)
            
            if not video_url:
                continue
                
            filename = f"video_part_{i}_{self.task_id}.mp4"
            local_path = os.path.join(TEMP_DIR, filename)
            xapi_headers = self.prepare_xapi_headers(video_info)
            
            print(f"Загрузка видео {i+1}: {video_url}")
            if self.downloader.download_video(video_url, local_path, xapi_headers):
                downloaded_files.append(local_path)
                print(f"Видео {i+1} успешно загружено: {local_path}")
            else:
                print(f"Ошибка загрузки видео {i+1}")
        
        if not downloaded_files:
            raise Exception("Не удалось загрузить ни одного видео")
            
        return self.combine_downloaded_videos(downloaded_files)

    def extract_video_url(self, video_info):
        """Извлечение URL видео из данных piapi"""
        if isinstance(video_info, dict):
            if 'url' in video_info:
                return video_info['url']
            
            if 'output' in video_info and 'url' in video_info['output']:
                return video_info['output']['url']
            
            if 'output' in video_info and 'works' in video_info['output']:
                try:
                    works = video_info['output']['works']
                    if isinstance(works, list) and works:
                        work = works[0]
                        if isinstance(work, dict) and 'video' in work:
                            video_data = work['video']
                            if 'resource_without_watermark' in video_data:
                                return video_data['resource_without_watermark']
                            elif 'resource' in video_data:
                                return video_data['resource']
                except Exception as e:
                    print(f"DEBUG [{self.task_id}]: Error processing works structure: {e}")
            
            for key, value in video_info.items():
                if isinstance(value, str) and value.startswith('http') and '.mp4' in value:
                    return value
        
        return None

    def prepare_xapi_headers(self, video_info):
        """Подготовка xAPI заголовков для загрузки"""
        xapi_headers = {
            'X-Experience-API-Version': '1.0.3',
            'X-Video-Source': 'piapi',
            'X-Video-Processing': 'automated',
        }
        
        if isinstance(video_info, dict):
            if 'task_id' in video_info:
                xapi_headers['X-Video-Task-ID'] = video_info['task_id']
            if 'model' in video_info:
                xapi_headers['X-Video-Model'] = video_info['model']
        
        return xapi_headers

    def combine_downloaded_videos(self, video_files):
        """Объединение загруженных видео файлов"""
        if not video_files:
            return None

        if len(video_files) == 1:
            combined_path = os.path.join(TEMP_DIR, f"combined_{self.task_id}.mp4")
            try:
                input_stream = ffmpeg.input(video_files[0])
                has_audio = self.check_audio_exists(video_files[0])
                
                output_args = [input_stream.video]
                output_kwargs = {'vcodec': 'libx264', 'preset': 'fast'}
                
                if has_audio:
                    output_args.append(input_stream.audio)
                    output_kwargs['acodec'] = 'aac'
                
                (ffmpeg
                 .output(*output_args, combined_path, **output_kwargs)
                 .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
                
                return combined_path
            except Exception as e:
                print(f"Задача {self.task_id}: Ошибка обработки единственного файла: {e}")
                raise
        
        # Объединение нескольких файлов
        inputs = []
        any_input_has_audio = False
        
        for video_file in video_files:
            if os.path.exists(video_file):
                inputs.append(ffmpeg.input(video_file))
                if not any_input_has_audio and self.check_audio_exists(video_file):
                    any_input_has_audio = True
        
        if not inputs:
            return None
        
        combined_path = os.path.join(TEMP_DIR, f"combined_{self.task_id}.mp4")
        
        concat_params = {'n': len(inputs), 'v': 1}
        if any_input_has_audio:
            concat_params['a'] = 1
        else:
            concat_params['a'] = 0
        
        try:
            joined_streams = ffmpeg.concat(*inputs, **concat_params)
            
            output_kwargs = {'vcodec': 'libx264', 'preset': 'fast'}
            if any_input_has_audio:
                output_kwargs['acodec'] = 'aac'
            
            (ffmpeg
             .output(joined_streams, combined_path, **output_kwargs)
             .run(overwrite_output=True, capture_stdout=True, capture_stderr=True))
            
            return combined_path
            
        except Exception as e:
            print(f"Задача {self.task_id}: Ошибка объединения видео: {e}")
            raise

    def extract_text_from_videos(self):
        """Извлечение текста из описаний видео для генерации субтитров"""
        full_text_parts = []
        for video_info in self.video_data:
            current_raw_text = ""
            if isinstance(video_info, dict):
                description = video_info.get('description', '')
                title = video_info.get('title', '')
                if description and title:
                    current_raw_text = f"{title}. {description}" 
                elif description:
                    current_raw_text = description
                elif title:
                    current_raw_text = title
            elif isinstance(video_info, str):
                current_raw_text = video_info
            
            if current_raw_text:
                cleaned_text = self.clean_text_for_speech(current_raw_text)
                if cleaned_text:
                    full_text_parts.append(cleaned_text)
        
        return "\n\n".join(full_text_parts)

    def clean_text_for_speech(self, text_input):
        """Очистка текста для субтитров"""
        if not text_input:
            return ""
        
        if not isinstance(text_input, str):
            try:
                text_input = str(text_input)
            except Exception:
                return ""
        
        processed_text = text_input
        processed_text = re.sub(r'https?://\S+', '', processed_text)
        
        tech_terms = ['POV', 'GoPro', 'MacBook', 'LinkedIn', 'TikTok', 'iPhone camera']
        for term in tech_terms:
            processed_text = re.sub(r'\b' + re.escape(term) + r'\b', '', processed_text, flags=re.IGNORECASE)
        
        processed_text = re.sub(r'realistic and casual.*', '', processed_text, flags=re.IGNORECASE)
        processed_text = re.sub(r'first person view.*?shot of', 'Видео показывает', processed_text, flags=re.IGNORECASE)
        
        processed_text = processed_text.strip()
        processed_text = re.sub(r'\s+', ' ', processed_text)
        
        return processed_text[:500]

    def generate_subtitles_from_text(self, text):
        """Генерация субтитров из текста в формате line_level"""
        words = text.split()
        subtitle_duration = 3
        words_per_subtitle = 6
        
        line_level_subtitles = []
        
        for i in range(0, len(words), words_per_subtitle):
            start_time = (i // words_per_subtitle) * subtitle_duration
            end_time = start_time + subtitle_duration
            
            subtitle_text = " ".join(words[i:i + words_per_subtitle])
            
            line_level_subtitles.append({
                'text': subtitle_text,
                'start': start_time,
                'end': end_time
            })
        
        return line_level_subtitles

# Flask routes без изменений
@app.route('/webhook', methods=['POST'])
def handle_webhook():
    """Обработчик webhook от n8n для piapi данных с поддержкой субтитров"""
    try:
        data = request.get_json()
        
        # Извлечение данных о видео
        video_list = []
        subtitles_content = None
        
        if not isinstance(data, dict):
            return jsonify({
                "error": "Неверный формат webhook: ожидается JSON объект",
                "status": "error"
            }), 400
        
        # Получение списка видео
        if 'videos' in data and isinstance(data['videos'], list):
            video_list = data['videos']
        elif 'data' in data and isinstance(data['data'], list):
            video_list = data['data']
        else:
            return jsonify({
                "error": "Неверный формат webhook: отсутствует массив 'videos' или 'data'",
                "status": "error"
            }), 400
        
        # Получение субтитров (опционально)
        if 'srt' in data:
            subtitles_content = data['srt']
            print(f"Получены субтитры для обработки: {len(subtitles_content) if subtitles_content else 0} символов")
        
        if not video_list:
            return jsonify({
                "error": "Массив видео пуст",
                "status": "error"
            }), 400
        
        # Генерация task_id
        task_id = str(uuid.uuid4())
        
        # Инициализация задачи
        TASKS[task_id] = {
            "status": "initiated",
            "progress": 0,
            "created_at": datetime.now().isoformat(),
            "video_data": video_list,
            "video_count": len(video_list),
            "has_custom_subtitles": bool(subtitles_content)
        }
        
        # Запуск обработки с субтитрами
        processor = VideoProcessor(task_id, video_list, subtitles_content)
        thread = threading.Thread(target=processor.process_videos)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            "task_id": task_id,
            "status": "processing",
            "message": f"Начата обработка {len(video_list)} видео с прокручивающимися субтитрами",
            "video_count": len(video_list),
            "has_custom_subtitles": bool(subtitles_content)
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Ошибка обработки webhook: {str(e)}",
            "status": "error"
        }), 500

@app.route('/status/<task_id>', methods=['GET'])
def get_task_status(task_id):
    if task_id not in TASKS:
        return jsonify({"error": "Задача не найдена"}), 404
    
    task_info = TASKS[task_id]
    response = {
        "task_id": task_id,
        "status": task_info["status"],
        "progress": task_info["progress"],
        "created_at": task_info["created_at"],
        "video_count": task_info.get("video_count", 0),
        "has_custom_subtitles": task_info.get("has_custom_subtitles", False)
    }
    
    if task_info["status"] == "completed":
        response["download_url"] = f"/download/{task_id}"
        response["output_file"] = task_info.get("output_file", "")
    
    if task_info["status"] == "error":
        response["error"] = task_info.get("error", "Неизвестная ошибка")
    
    if "warnings" in task_info:
        response["warnings"] = task_info["warnings"]
    
    return jsonify(response)

@app.route('/download/<task_id>', methods=['GET'])
def download_video(task_id):
    if task_id not in TASKS:
        return jsonify({"error": "Задача не найдена"}), 404
    
    task_info = TASKS[task_id]
    if task_info["status"] != "completed":
        return jsonify({"error": "Видео еще не готово"}), 400
    
    output_file = task_info.get("output_file", "")
    if not os.path.exists(output_file):
        return jsonify({"error": "Файл не найден"}), 404
    
    return send_file(
        output_file,
        as_attachment=True,
        download_name=f"final_scrolling_subtitles_{task_id}.mp4"
    )

if __name__ == '__main__':
    print("🎬 Запуск сервера обработки видео с прокручивающимися субтитрами...")
    print("📡 Webhook endpoint: http://localhost:9000/webhook")
    print("📊 Status endpoint: http://localhost:9000/status/<task_id>")
    print("⬇️ Download endpoint: http://localhost:9000/download/<task_id>")
    print("\n📝 Формат webhook с субтитрами:")
    print('{"videos": [...], "srt": "1\\n00:00:00,000 --> 00:00:03,000\\nПервая строка субтитров\\n\\n2\\n00:00:03,000 --> 00:00:06,000\\nВторая строка субтитров"}')
    
    # Установка зависимостей
    print("\n🔧 Для полной функциональности установите:")
    print("pip install faster-whisper openai-whisper moviepy flask ffmpeg-python pysrt requests")
    
    app.run(host='0.0.0.0', port=9000, debug=True)
