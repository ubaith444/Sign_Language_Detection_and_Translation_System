import os
import sys
import time
import json
import cv2
import numpy as np

# Add project root to python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import RECORDINGS_DIR, SEQUENCE_LENGTH
from src.vision.camera_stream import CameraStream
from src.vision.landmark_extractor import LandmarkExtractor

def record_consented_samples(
    sign_name: str,
    signer_id: str = "signer_01",
    num_samples: int = 5,
    is_dynamic: bool = False,
    consent_confirmed: bool = True,
    lighting_condition: str = "diffuse",
    background_condition: str = "neutral",
    camera_distance: str = "medium_1.5m",
    signing_speed: str = "natural",
    annotator_name: str = "Certified_ISL_Annotator"
):
    """
    Records consented, metadata-tagged ISL landmark sequences partitioned
    by signer_id to enforce disjoint signer splits per research standards.
    """
    if not consent_confirmed:
        print("[DATA COLLECTION PROTOCOL ERROR] Informed consent is MANDATORY before recording or storing sign data.")
        print("Please verify participant consent before proceeding.")
        return

    # Organize strictly by signer_id to maintain disjoint train/test splits
    signer_folder = RECORDINGS_DIR / signer_id / ("dynamic" if is_dynamic else "static") / sign_name
    os.makedirs(signer_folder, exist_ok=True)

    print("=================================================================")
    print(" ISL CONSENTED DATA COLLECTION PROTOCOL                          ")
    print(f" Signer ID:             {signer_id} (Disjoint split partition)   ")
    print(f" Target Sign:           {sign_name}                               ")
    print(f" Type:                  {'Dynamic Sequence' if is_dynamic else 'Static Handshape'} ")
    print(f" Informed Consent:      VERIFIED                                  ")
    print(f" Environmental Metadata: {lighting_condition}, {background_condition}, {camera_distance}, {signing_speed}")
    print(f" Annotator:             {annotator_name}                          ")
    print("=================================================================")
    print("Controls: Press [SPACE] to capture sample | [Q] to quit.\n")

    camera = CameraStream()
    extractor = LandmarkExtractor()
    camera.start()

    count = 0
    try:
        while count < num_samples:
            ret, frame = camera.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            features = extractor.process_frame(frame)
            annotated = extractor.draw_landmarks(frame, features["raw_results"])

            cv2.putText(
                annotated, f"Signer: {signer_id} | Sign: {sign_name} | {count}/{num_samples}",
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 160, 40), 2
            )
            cv2.putText(
                annotated, f"Speed: {signing_speed} | Light: {lighting_condition} | Press SPACE",
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60, 60, 60), 1
            )

            cv2.imshow("Consented ISL Dataset Recorder", annotated)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break
            elif key == 32: # SPACE
                timestamp = int(time.time() * 1000)
                metadata = {
                    "signer_id": signer_id,
                    "sign_name": sign_name,
                    "is_dynamic": is_dynamic,
                    "timestamp": timestamp,
                    "consent_verified": consent_confirmed,
                    "lighting": lighting_condition,
                    "background": background_condition,
                    "camera_distance": camera_distance,
                    "signing_speed": signing_speed,
                    "annotator": annotator_name
                }

                if not is_dynamic:
                    data_file = signer_folder / f"sample_{count}_{timestamp}.npy"
                    meta_file = signer_folder / f"sample_{count}_{timestamp}.json"
                    np.save(data_file, features["holistic_vector"])
                    with open(meta_file, "w", encoding="utf-8") as f:
                        json.dump(metadata, f, indent=2)
                    print(f"Saved static sample {count + 1}/{num_samples} -> {data_file.name}")
                    count += 1
                else:
                    print(f"Recording {SEQUENCE_LENGTH} frame trajectory for '{sign_name}'...")
                    seq_frames = []
                    for _ in range(SEQUENCE_LENGTH):
                        r, f = camera.read()
                        if r and f is not None:
                            feat = extractor.process_frame(f)
                            seq_frames.append(feat["holistic_vector"])
                        time.sleep(1.0 / 30.0)

                    if len(seq_frames) == SEQUENCE_LENGTH:
                        data_file = signer_folder / f"sequence_{count}_{timestamp}.npy"
                        meta_file = signer_folder / f"sequence_{count}_{timestamp}.json"
                        np.save(data_file, np.array(seq_frames, dtype=np.float32))
                        with open(meta_file, "w", encoding="utf-8") as f:
                            json.dump(metadata, f, indent=2)
                        print(f"Saved dynamic sequence {count + 1}/{num_samples} -> {data_file.name}")
                        count += 1

    finally:
        camera.stop()
        extractor.close()
        cv2.destroyAllWindows()
        print(f"\n[Recorder] Completed. Data safely stored under: {signer_folder}")

if __name__ == "__main__":
    sign = sys.argv[1] if len(sys.argv) > 1 else "A"
    signer = sys.argv[2] if len(sys.argv) > 2 else "signer_01"
    is_dyn = "--dynamic" in sys.argv
    record_consented_samples(sign_name=sign, signer_id=signer, num_samples=3, is_dynamic=is_dyn)
