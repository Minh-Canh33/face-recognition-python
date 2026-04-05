import cv2

def draw_results(frame, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        # scale lại vì detect ở 0.5
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        cv2.putText(frame, name, (left+10, top-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

    return frame
