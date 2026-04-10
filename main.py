import time

import cv2
from SOPVerifier import SOPVerifier
from DINOv2Encoder import  DINOv2Encoder
from SOPReferenceBank import SOPReferenceBank
import os
from config import load_config, AppConfig


def run_live(verifier: SOPVerifier, capture_key='s', quit_key='q'):
    """
    Press `s` to capture and verify the current frame.
    Press `q` to quit.
    """
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,2160)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)

    print(f"\nStarting. Expecting step {verifier.current_step}: "
          f"'{verifier.bank.steps[verifier.current_step]['step_name']}'")

    while True:
        ret, frame = cap.read()
        resize_frame = cv2.resize(frame.copy(), (640, 480))
        if not ret:
            break

        # Overlay current expected step on screen
        step_name = verifier.bank.steps[verifier.current_step]['step_name']
        cv2.putText(resize_frame, f"Expected: {step_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(resize_frame, f"Press '{capture_key}' to verify", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow('SOP Verifier', resize_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(capture_key):
            cv2.imwrite(f"capture_{step_name}.png", frame.copy())
            result = verifier.verify(frame)
            color = (0, 255, 0) if result['passed'] else (0, 0, 255)
            print(f"\n{result['message']}")
            print(f"  Similarity: {result['similarity']}")

            # Flash result on screen for 1.5s
            overlay = resize_frame.copy()
            cv2.rectangle(overlay, (0, 0), (640, 80), color, -1)
            cv2.addWeighted(overlay, 0.4, resize_frame, 0.6, 0, resize_frame)
            cv2.putText(resize_frame, result['message'][:60], (10, 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow('SOP Verifier', resize_frame)
            cv2.waitKey(1500)

        elif key == ord(quit_key):
            break

    cap.release()
    cv2.destroyAllWindows()

mouse_position = None
def mouse_callback(event, x, y, flags, param):
    global mouse_position
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"Mouse clicked: ({x}, {y})")
        mouse_position = (x, y)




def inference_dino_single_img():
    encoder = DINOv2Encoder(model_name='dinov2_vitb14')

    bank = SOPReferenceBank(encoder)
    bank.register_from_folder("data/capture/SOP_IMAGE/")
    bank.save('data/capture/SOP_IMAGE/RESULTS_ENCODE/')

    # Load existing bank
    # bank = SOPReferenceBank(encoder)
    # bank.load('data/capture/SOP_IMAGE/RESULTS_ENCODE/')

    # Create verifier
    verifier = SOPVerifier(
        encoder=encoder,
        bank=bank,
        pass_threshold=0.7
    )
    verifier.current_step = 0  # Start with STEP 1

    # Define cropping coordinates for each step
    step_crops = {
        0: {'coords': (73, 228, 170, 285), 'name': 'STEP1'},  # Y1, Y2, X1, X2
        # 1: {'coords': (5, 186, 167, 459), 'name': 'STEP2'},
        # 1: {'coords': (41, 155, 285, 394), 'name': 'STEP2'},
        # 2: {'coords': (5, 151, 180, 420), 'name': 'STEP3'}  # Add different coords for STEP3 if needed
        1: {'coords': (5, 151, 180, 420), 'name': 'STEP3'}  # Add different coords for STEP3 if needed
    }

    # Track completed steps
    completed_steps = set()

    # Folder containing frames
    frames_folder = "data/capture/frames"
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')

    # Get sorted list of frame files
    frame_files = [
        f for f in os.listdir(frames_folder)
        if f.lower().endswith(valid_extensions)
    ]
    frame_files.sort()  # Sort alphabetically for consistent order

    if not frame_files:
        print(f"No valid images found in {frames_folder}")
        return

    current_index = 0

    def draw_step_status(img, completed_steps):
        """Draw step completion status on the image"""
        height, width = img.shape[:2]
        y_offset = 20
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1

        for step_idx in range(len(step_crops)):
            step_name = step_crops[step_idx]['name']
            color = (0, 255, 0) if step_idx in completed_steps else (0, 0, 255)  # Green if done, Red if pending
            status = "DONE" if step_idx in completed_steps else "PENDING"

            text = f"{step_name}: {status}"
            cv2.putText(img, text, (10, y_offset), font, font_scale, color, thickness)
            y_offset += 25
            # Show selected zone info
        if mouse_position:
                zone_text = f"Selected Zone: ({mouse_position[0]}, {mouse_position[1]})"
                cv2.putText(img, zone_text, (10, height - 20), font, font_scale, (255, 255, 255), thickness)

        # Show current step
        current_step_text = f"CURRENT: {step_crops[verifier.current_step]['name']}"
        cv2.putText(img, current_step_text, (width - 200, 20), font, font_scale, (255, 255, 0), thickness)

        return img

    def process_frame(frame_path):
        """Process a single frame with dynamic cropping based on current step"""
        step_name = step_crops[verifier.current_step]['name']
        cv2.namedWindow(f"Original Frame - {step_name}")
        cv2.setMouseCallback(f"Original Frame - {step_name}", mouse_callback)

        img_test = cv2.imread(frame_path)
        if img_test is None:
            print(f"Could not read image: {frame_path}")
            return False

        # Resize image
        resize_img = cv2.resize(img_test, (640, 480))

        # Draw step status on original image
        status_img = draw_step_status(resize_img.copy(), completed_steps)

        # Get cropping coordinates for current step
        if verifier.current_step not in step_crops:
            print(f"Step {verifier.current_step} not defined in step_crops")
            return False

        y1, y2, x1, x2 = step_crops[verifier.current_step]['coords']
        crop_img = resize_img[y1:y2, x1:x2]

        # Display images with step information

        cv2.imshow(f"Original Frame - {step_name}", status_img)
        cv2.imshow(f"Cropped Region - {step_name}", crop_img)

        # # Save cropped image with step info
        cv2.imwrite(f"STEP_{verifier.current_step}.png", crop_img)

        # Perform verification
        start_time = time.time()
        result = verifier.verify(crop_img)
        end_time = time.time()
        inference_time = end_time - start_time

        print(f"Frame: {os.path.basename(frame_path)}")
        print(f"Current Step: {step_name}")
        print(f"Result: {result}")
        print(f"Inference time: {inference_time:.4f}s")
        print("-" * 40)

        # If verification passes, advance to next step
        if result["passed"]:
            completed_steps.add(verifier.current_step)
            if verifier.current_step < len(step_crops) - 1:
                # print(f"before current_step: {verifier.current_step}")
                verifier.current_step += 1
                print(f"next current step {verifier.current_step}")
                print(f"Advancing to next step: {step_crops[verifier.current_step]['name']}")
            else:
                verifier.current_step  = 0
                print("All steps completed successfully!, Reset")
                # Optionally reset to step 0 after all steps complete
                # verifier.current_step = 0
                # completed_steps.clear()

        return result

    # Process first frame
    first_frame_path = os.path.join(frames_folder, frame_files[current_index])
    verifier.current_step = 0
    process_frame(first_frame_path)

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key == ord('q') or key == 27:  # 'q' or ESC to quit
            break
        elif key == ord('n') or key == ord(' ') or key == 83:  # 'n', space, or right arrow for next
            current_index = (current_index + 1) % len(frame_files)  # Loop back to start
            frame_path = os.path.join(frames_folder, frame_files[current_index])
            process_frame(frame_path)
        elif key == ord('p') or key == 81:  # 'p' or left arrow for previous
            current_index = (current_index - 1) % len(frame_files)  # Loop back to end
            frame_path = os.path.join(frames_folder, frame_files[current_index])
            process_frame(frame_path)
        elif key == ord('r'):  # 'r' to retry current step on same frame
            verifier.current_step = 0
            frame_path = os.path.join(frames_folder, frame_files[current_index])
            process_frame(frame_path)
        elif key == ord('z'):  # 'z' to save selected zone as crop
                # Get the current frame again for cropping
                img_test = cv2.imread(os.path.join(frames_folder, frame_files[current_index]))
                if img_test is not None:
                    resize_img = cv2.resize(img_test, (640, 480))
                    y1, y2, x1, x2 = step_crops[verifier.current_step]['coords']
                    zone_crop = resize_img[y1:y2, x1:x2]

                    # Save the selected zone
                    zone_filename = f"ZONE_STEP_{verifier.current_step}.png"
                    cv2.imwrite(zone_filename, zone_crop)
                    print(f"Zone crop saved as: {zone_filename}")
                    print(f"Cropped coordinates: ({x1}, {y1}) to ({x2}, {y2})")


    cv2.destroyAllWindows()


from FeatureBasedVerifier import FeatureBasedVerifier  # Add this import


def build_feature_verifiers(cfg: AppConfig) -> dict[int, FeatureBasedVerifier]:
    """Builds FeatureBasedVerifier instances based on inspect_config in config."""
    feature_verifiers = {}
    for step in cfg.sop_steps:
        if step.needs_inspect:
            # You can configure the verifier parameters per step or use defaults
            # Here, using parameters from config or defaults
            # Pass the specific folder for this step
            ref_folders = {step.step_id: step.inspect_config.reference_folder}

            # Use step-specific pass_threshold if defined, otherwise use a default
            step_pass_thresh = step.inspect_config.pass_threshold
            if step_pass_thresh is None:
                # Fallback to a default value or cfg.dino.pass_threshold if applicable
                step_pass_thresh = 0.6  # Example default, adjust as needed

            feature_verifiers[step.step_id] = FeatureBasedVerifier(
                feature_extractor='SIFT',  # Or 'ORB'
                matcher_type='BF',  # Or 'BF'
                match_ratio_threshold=0.75,
                min_matches_needed=10,
                pass_threshold=step_pass_thresh,  # Use the threshold from config
                reference_folders=ref_folders
            )
    return feature_verifiers



def run_sop_logic_zone():
    from config import load_config
    from hand_tracker import HandTracker, HandState
    from sop_engine import SOPEngine
    from renderer import Renderer

    cfg = load_config("config.yaml")
    verifiers = build_verifiers(cfg)
    feature_verifiers = build_feature_verifiers(cfg) # Build new feature verifiers

    engine = SOPEngine(cfg, verifiers=verifiers, feature_verifiers=feature_verifiers)
    renderer = Renderer(cfg)

    cap = cv2.VideoCapture(cfg.camera.source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.frame_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.frame_h)
    cap.set(cv2.CAP_PROP_FPS, cfg.camera.fps)

    fps, prev_t = 0.0, time.time()
    cv2.namedWindow("SOP Assembly")
    cv2.setMouseCallback("SOP Assembly", mouse_callback)

    with HandTracker(cfg) as tracker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            display = cv2.resize(frame, (cfg.camera.frame_w, cfg.camera.frame_h))

            # ── Hand detection ─────────────────────────────────────────────────
            hand = HandState()
            if not engine.all_done:
                hand = tracker.process(frame, display, engine.current_step)

            # ── SOP state machine ──────────────────────────────────────────────
            flash = engine.update(hand, 0,frame=display)

            # ── Render overlays ────────────────────────────────────────────────
            fps = 0.9 * fps + 0.1 / max(time.time() - prev_t, 1e-6)
            prev_t = time.time()
            renderer.draw_frame(display, engine, hand, flash, fps)

            cv2.imshow("SOP Assembly", display)

            # ── Hotkeys ────────────────────────────────────────────────────────
            key = cv2.waitKey(10) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                fname = f"snap_{int(time.time())}.png"
                cv2.imwrite(fname, display)
                print(f"[SNAP] Saved {fname}")
            elif key == ord("p"):
                print("Paused — press any key to continue")
                cv2.waitKey(0)
            elif key == ord("r"):
                engine.reset()
                print("[HOTKEY] Manual reset")

    cap.release()
    cv2.destroyAllWindows()


def build_verifiers(cfg: AppConfig) -> dict:
    """
    For every inspect-mode step, create a SOPVerifier backed by its own
    SOPReferenceBank loaded from the step's reference_folder.

    Returns a dict  { step_id: SOPVerifier }
    Only imports DINOv2 / torch if there are actually inspect steps (lazy load).
    """
    inspect_steps = [s for s in cfg.sop_steps if s.needs_inspect]
    if not inspect_steps:
        print("[INIT] No inspect steps — skipping DINOv2 load.")
        return {}

    print("[INIT] Inspect steps detected — loading DINOv2 encoder...")
    from DINOv2Encoder import DINOv2Encoder
    from SOPReferenceBank import SOPReferenceBank
    from SOPVerifier import SOPVerifier

    encoder = DINOv2Encoder(model_name=cfg.dino.model_name)
    verifiers: dict = {}

    for step in inspect_steps:
        ins = step.inspect_config
        threshold = ins.pass_threshold if ins.pass_threshold is not None \
            else cfg.dino.pass_threshold

        print(f"  → Building reference bank for '{step.name}' "
              f"from '{ins.reference_folder}' (threshold={threshold:.2f})")

        bank = SOPReferenceBank(encoder)

        # Use cached embeddings if available — skip slow re-encoding
        embed_path = ins.reference_folder / "embeddings"
        meta_path = str(embed_path) + "_metadata.json"
        npy_path = str(embed_path) + "_embeddings.npy"

        if os.path.exists(meta_path) and os.path.exists(npy_path):
            print(f"    Loading cached embeddings from {embed_path}")
            bank.load(str(embed_path))
        else:
            image_files = [
                f for f in os.listdir(str(ins.reference_folder))
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            if not image_files:
                print(f"    [WARN] No images found in '{ins.reference_folder}'. "
                      f"Step {step.name} inspect will always fail. "
                      f"Add reference images and restart.")
                continue
            bank.register_from_folder(str(ins.reference_folder))
            bank.save(str(embed_path))

        verifiers[step.step_id] = SOPVerifier(
            encoder=encoder,
            bank=bank,
            pass_threshold=threshold,
        )

    print(f"[INIT] {len(verifiers)} verifier(s) ready.\n")
    return verifiers


if __name__ == '__main__':

    run_sop_logic_zone()
    # inference_dino_single_img()


    # # 1. Init encoder
    # encoder = DINOv2Encoder(model_name='dinov2_vitb14')
    #
    # # 2. Build reference bank (run once)
    # bank = SOPReferenceBank(encoder)
    # bank.register_from_folder("image_test/SOP")
    # bank.save('image_test/SOP')
    #
    # # --- OR load existing bank ---
    # # bank = SOPReferenceBank(encoder)
    # # bank.load('image_test/SOP')
    #
    # # 3. Create verifier
    # verifier = SOPVerifier(
    #     encoder=encoder,
    #     bank=bank,
    #     pass_threshold=0.8   # tune this per your use case
    # )
    # #
    # # # 4. Run live
    # run_live(verifier)