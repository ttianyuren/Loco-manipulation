import threading
import keyboard
import time
import os
import argparse

from sound_recorder import SoundRecorder
from robot_recorder import RobotRecorder
from whisper_transcriber import WhisperTranscriber
from api_keyword_extractor import GPTTextProcessor

def main():
    
    # initialize SoundRecorder
    sound_recorder = SoundRecorder(folder="sound_storage")

    # initialize RobotRecorder
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--directory",
        type=str,
        default="./trajectory_storage"),
    parser.add_argument("--environment", type=str, default="Lift")
    parser.add_argument(
        "--robots",
        nargs="+",
        type=str,
        default="Panda",
        help="Which robot(s) to use in the env",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Specified environment configuration if necessary",
    )
    parser.add_argument(
        "--arm",
        type=str,
        default="right",
        help="Which arm to control (eg bimanual) 'right' or 'left'",
    )
    parser.add_argument(
        "--camera",
        nargs="*",
        type=str,
        default="agentview",
        help="List of camera names to use for collecting demos. Pass multiple names to enable multiple views. Note: the `mujoco` renderer must be enabled when using multiple views; `mjviewer` is not supported.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default=None,
        help="Choice of controller. Can be generic (eg. 'BASIC' or 'WHOLE_BODY_MINK_IK') or json file (see robosuite/controllers/config for examples)",
    )
    parser.add_argument("--device", type=str, default="keyboard")
    parser.add_argument(
        "--pos-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale position user inputs",
    )
    parser.add_argument(
        "--rot-sensitivity",
        type=float,
        default=1.0,
        help="How much to scale rotation user inputs",
    )
    parser.add_argument(
        "--renderer",
        type=str,
        default="mjviewer",
        help="Use Mujoco's builtin interactive viewer (mjviewer) or OpenCV viewer (mujoco)",
    )
    parser.add_argument(
        "--max_fr",
        default=20,
        type=int,
        help="Sleep when simluation runs faster than specified frame rate; 20 fps is real time.",
    )
    parser.add_argument(
        "--reverse_xy",
        type=bool,
        default=False,
        help="(DualSense Only)Reverse the effect of the x and y axes of the joystick.It is used to handle the case that the left/right and front/back sides of the view are opposite to the LX and LY of the joystick(Push LX up but the robot move left in your view)",
    )
    args = parser.parse_args()

    config = vars(args) #change argparse.Namespace into directionary
    robot_recorder = RobotRecorder(config)

    # Step 1: Wait for pressing "s"
    print("Press s to start recording sound and trajectory")
    keyboard.wait('s')
    print("s key detected, start synchronized recording!")

    # Step 2: start recording sound
    audio_thread = threading.Thread(target=sound_recorder.record_audio)
    audio_thread.start()

    # Step 3: start recording trajectory (main thread)
    # It is recommended to run the main thread directly and automatically stop the audio after recording is completed.
    robot_recorder.collect_trajectory()

    # Step 4: stop recording voice
    print("Robot arm recording is finished, stopping recording...")
    sound_recorder.is_recording = False
    audio_thread.join()

    print("Audio and trajectory recording completed\n")
    print("Start transcribing ...")

    # Step 5: Whisper transcribing
    transcriber = WhisperTranscriber(model_size="base")
    audio_path = transcriber.find_latest_wav(folder="sound_storage")
    result = transcriber.transcribe(audio_path)
    transcriber.print_word_timestamps(result)

    # Step 6: Extract motion units
    print("\nExtract motion units ...")
    extractor = GPTTextProcessor()

    print("Motion units are as follows")
    processed_result = extractor.process_text(result["text"])
    print(processed_result)

if __name__ == "__main__":
    main()