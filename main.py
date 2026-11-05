"""VoxWear real-time assistant: gesture, speech, environment and agent threads + TTS output.

Keys in the video window:  Q = quit,  E = toggle environment detection.
"""
import argparse
import queue
import threading
import time

import cv2

from voxwear.agent import VoxWearAgent
from voxwear.gesture import GestureRecognizer


def speech_worker(inbox: queue.Queue, stop: threading.Event, model_size: str):
    from voxwear.speech import SpeechToText
    stt = SpeechToText(model_size)
    while not stop.is_set():
        text = stt.transcribe(stt.record(3.0))
        if text:
            inbox.put(("speech", text))


def agent_worker(inbox: queue.Queue, outbox: queue.Queue, agent: VoxWearAgent, stop: threading.Event):
    while not stop.is_set():
        try:
            kind, payload = inbox.get(timeout=0.5)
        except queue.Empty:
            continue
        try:
            reply = agent.respond(gesture=payload) if kind == "gesture" else agent.respond(speech=payload)
        except Exception as e:  # keep running on network / model errors
            reply = f"(agent error: {e})"
        label = payload["gesture"] if kind == "gesture" else payload
        print(f"[{time.strftime('%H:%M:%S')}] User ({kind}): {label}\n"
              f"[{time.strftime('%H:%M:%S')}] VoxWear: {reply}")
        outbox.put(reply)


def tts_worker(outbox: queue.Queue, stop: threading.Event):
    from voxwear.speech import TextToSpeech
    tts = TextToSpeech()
    while not stop.is_set():
        try:
            tts.say(outbox.get(timeout=0.5))
        except queue.Empty:
            continue


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/gesture_xgb.joblib")
    ap.add_argument("--llm", default="openai:gpt-4o-mini")
    ap.add_argument("--whisper", default="base")
    ap.add_argument("--no-speech", action="store_true")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    stop = threading.Event()
    inbox, outbox = queue.Queue(maxsize=10), queue.Queue(maxsize=10)
    agent = VoxWearAgent(args.llm)
    recognizer = GestureRecognizer(args.model)

    threads = [threading.Thread(target=agent_worker, args=(inbox, outbox, agent, stop), daemon=True),
               threading.Thread(target=tts_worker, args=(outbox, stop), daemon=True)]
    if not args.no_speech:
        threads.append(threading.Thread(target=speech_worker, args=(inbox, stop, args.whisper),
                                        daemon=True))
    for t in threads:
        t.start()

    env, env_on, last_env, last_gesture, last_sent = None, False, 0.0, None, 0.0
    cap = cv2.VideoCapture(args.camera)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        g = recognizer.predict(frame)
        status = "No gesture"
        if g:
            recognizer.tracker.draw(frame, g["landmarks"])
            status = f"{g['gesture'].upper()} {g['confidence']:.0%} ({g['intent']})"
            # send to agent at most every 2 s, and only when the gesture changes
            if g["gesture"] != last_gesture and time.time() - last_sent > 2 and not inbox.full():
                inbox.put(("gesture", {k: v for k, v in g.items() if k != "landmarks"}))
                last_gesture, last_sent = g["gesture"], time.time()
        if env_on and time.time() - last_env > 5:
            if env is None:
                from voxwear.environment import EnvironmentDetector
                env = EnvironmentDetector()
            agent.update_environment(env.describe(frame))
            last_env = time.time()

        cv2.putText(frame, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Env: {'ON' if env_on else 'OFF'}", (15, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("VoxWear", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("e"):
            env_on = not env_on

    stop.set()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
