import sys
import cv2

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor
from vision.utils.misc import Timer


def net_select(model_type, class_names):
    if model_type == 'vgg16-ssd':
        return create_vgg_ssd(len(class_names), is_test=True)
    elif model_type == 'mb1-ssd':
        return create_mobilenetv1_ssd(len(class_names), is_test=True)
    elif model_type == 'mb1-ssd-lite':
        return create_mobilenetv1_ssd_lite(len(class_names), is_test=True)
    elif model_type == 'mb2-ssd-lite':
        return create_mobilenetv2_ssd_lite(len(class_names), is_test=True)
    elif model_type == 'sq-ssd-lite':
        return create_squeezenet_ssd_lite(len(class_names), is_test=True)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)
        
        
def create_predictor(model, model_type):
    if model_type == 'vgg16-ssd':
        return create_vgg_ssd_predictor(model, candidate_size=200)
    elif model_type == 'mb1-ssd':
        return create_mobilenetv1_ssd_predictor(model, candidate_size=200)
    elif model_type == 'mb1-ssd-lite':
        return create_mobilenetv1_ssd_lite_predictor(model, candidate_size=200)
    elif model_type == 'mb2-ssd-lite':
        return create_mobilenetv2_ssd_lite_predictor(model, candidate_size=200)
    elif model_type == 'sq-ssd-lite':
        return create_squeezenet_ssd_lite_predictor(model, candidate_size=200)
    else:
        print("The net type is wrong. It should be one of vgg16-ssd, mb1-ssd and mb1-ssd-lite.")
        sys.exit(1)


def main():
    if len(sys.argv) < 4:
        print('Usage: python run_ssd_example.py <net type> <model path> <label path> [video file]')
        sys.exit(0)

    net_type = sys.argv[1]
    model_path = sys.argv[2]
    label_path = sys.argv[3]
    if len(sys.argv) >= 5:
        cap = cv2.VideoCapture(sys.argv[4])  # capture from file
    else:
        cap = cv2.VideoCapture(0)  # capture from camera
        cap.set(3, 1920)
        cap.set(4, 1080)

    class_names = [name.strip() for name in open(label_path).readlines()]
    net = net_select(model_type=net_type, class_names=class_names)
    predictor = create_predictor(model=net, model_type=net_type)
    net.load(model_path)

    count = 0
    max_count = 10
    fps = 0
    timer = Timer()
    tm = cv2.TickMeter()
    tm.start()
    while True:
        ret, orig_image = cap.read()
        if orig_image is None:
            continue

        # for fps count
        if count == max_count:
            tm.stop()
            fps = max_count / tm.getTimeSec()
            tm.reset()
            tm.start()
            count = 0

        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        timer.start()
        boxes, labels, probs = predictor.predict(image, 10, 0.4)
        interval = timer.end()
        print('Time: {:.3f}s, Detect Objects: {:d}.'.format(interval, labels.size(0)))

        # fpsの書き込み
        cv2.putText(orig_image, f'{fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        for i in range(boxes.size(0)):
            # ボックスの書き込み
            box = boxes[i, :]
            label = f"{class_names[labels[i]]}: {probs[i]:.3f}"
            # ラベルと確信度の書き込み
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(orig_image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        cv2.imshow('Capture Demo', orig_image)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
