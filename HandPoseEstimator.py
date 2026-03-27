import cv2
import mediapipe as mp
import numpy as np

class HandPoseEstimator:

    def __init__(self, max_hands: int = 2, min_detect_conf: float = 0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            model_complexity=1,
            max_num_hands=max_hands,
            min_detection_confidence=min_detect_conf,
            min_tracking_confidence=0.5,
        )

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_draw_style = mp.solutions.drawing_styles

        self.custom_conn_style = self.mp_drawing.DrawingSpec(color=(0,210,255), thickness=2)
        self.custom_lm_style = self.mp_drawing.DrawingSpec(color=(50,220,290), thickness=4,
                                                 circle_radius=4)

        self.results = None

    def process(self, image_rgb: np.ndarray):
        self.results = self.hands.process(image_rgb)


    def get_all_hand_features(self):
        if not self.results or not self.results.multi_hand_landmarks:
            return []

        all_hands = []

        for i, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
            label = self.results.multi_handedness[i].classification[0].label

            # convert ke numpy array (21,2)
            points = []
            for lm in hand_landmarks.landmark:
                points.append([lm.x, lm.y])
            points = np.array(points)

            # normalize
            norm_points = self.normalize_hand(points, label)

            # flatten untuk embedding
            feature = norm_points.flatten()

            all_hands.append({
                "label": label,
                "feature": feature,
            })

        return all_hands

    def draw_landmarks_visual(self, frame):
        for lms in self.results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(frame,
                                           lms,
                                           self.mp_hands.HAND_CONNECTIONS,
                                           self.custom_lm_style,
                                           self.custom_conn_style
                                           )


    def normalize_hand(self, points, label):
        """
        points: (21,2)
        """

        p0 = points[0]   # wrist
        p5 = points[5]   # index_mcp
        p17 = points[17] # pinky_mcp
        p9 = points[9]   # middle_mcp

        # 1. CENTER (palm center)
        center = (p0 + p5 + p17) / 3
        points = points - center

        # 2. SCALE (normalize size)
        scale = np.linalg.norm(p9 - p0)
        if scale > 1e-6:
            points = points / scale

        # 3. ROTATION (align direction)
        vec = p5 - p0
        angle = np.arctan2(vec[1], vec[0])

        R = np.array([
            [np.cos(-angle), -np.sin(-angle)],
            [np.sin(-angle),  np.cos(-angle)]
        ])

        points = np.dot(points, R.T)

        return points


if __name__ == "__main__":
    image_1 = cv2.imread("image_test/SOP/STEP_2.png")
    img_rgb = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)

    hand_pose = HandPoseEstimator()

    hand_pose.process(img_rgb)
    hand_pose.draw_landmarks_visual(img_rgb)
    hand_features = hand_pose.get_all_hand_features()
    print(hand_features)
    print(hand_features[0]['feature'].shape)
    cv2.imshow("Visual", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
