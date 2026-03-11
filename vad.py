import torch
import sounddevice as sd
import numpy as np
import soundfile as sf
import time

"""
- conda create -n kdt python=3.11 (신규 생성)
- conda env list (env list 확인)

[txt 파일에 리스트 업 된 라이브러리 한번에 설치하는 방법]
- pip install -r requirement.txt
"""

model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad',
    model='silero_vad'
)

(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

vad_iterator = VADIterator(model)


SAMPLING_RATE = 16000
BLOCK_SIZE = 512
RECORD_SECONDS = 10

speech_buffer = []
is_speaking = False

print('녹음 시작')

start_time = time.time()

with sd.InputStream(
    samplerate=SAMPLING_RATE, channels=1, blocksize=BLOCK_SIZE) as stream:
    
    while True:
        audio_chunk, _ = stream.read(BLOCK_SIZE)
        audio_chunk = audio_chunk.flatten()

        audio_tensor = torch.from_numpy(audio_chunk)

        speech_dict = vad_iterator(audio_tensor)

        if speech_dict:

            if 'start' in speech_dict:
                is_speaking = True
                print('speech start')

            if 'end' in speech_dict:
                is_speaking = False
                print('speech end')

            if is_speaking:
                speech_buffer.extend(audio_chunk)

            if time.time() - start_time > RECORD_SECONDS:
                break

print('녹음 완료')

speech_audio = np.array(speech_buffer)

sf.write(
    'vad_recorded.wav', speech_audio, SAMPLING_RATE)

print('파일 저장 완료')