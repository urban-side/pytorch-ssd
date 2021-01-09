import sys
import os
import cv2

from vision.ssd.vgg_ssd import create_vgg_ssd, create_vgg_ssd_predictor
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite, create_mobilenetv1_ssd_lite_predictor
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite, create_squeezenet_ssd_lite_predictor
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite, create_mobilenetv2_ssd_lite_predictor


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
        print('Usage: python run_ssd_example.py <net type> <model path> <label path> [image path]')
        sys.exit(0)

    net_type = sys.argv[1]
    model_path = sys.argv[2]
    label_path = sys.argv[3]
    if len(sys.argv) >= 5:
        image_path = sys.argv[4]
        image_path_list = [image_path]      # [path/to/example.jpg]
    else:
        image_dir = "./sample_data/"
        image_list = os.listdir(image_dir)  # [example.jpg, example1.png, ...]
        image_path_list = [image_dir + image_path for image_path in image_list]    # [path/to/example.jpg, ...]

    class_names = [name.strip() for name in open(label_path).readlines()]
    net = net_select(model_type=net_type, class_names=class_names)
    predictor = create_predictor(model=net, model_type=net_type)
    net.load(model_path)

    pics_index = 0
    for image_path in image_path_list:
        orig_image = cv2.imread(image_path)
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

        boxes, labels, probs = predictor.predict(image=image, top_k=10, prob_threshold=0.5)

        for i in range(boxes.size(0)):
            # ボックスの書き込み
            box = boxes[i, :]
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            # ラベルと確信度の書き込み
            label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
            cv2.putText(orig_image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type

        # 結果の画像はそれぞれ保存
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        save_path = 'sample_data/detected_result/' + image_name + '_res.jpg'
        cv2.imwrite(filename=save_path, img=orig_image)
        print('Saved picture: ', save_path)

        # 最後の1枚のみデモとして表示
        if pics_index == len(image_path_list) - 1:
            cv2.imshow('Predicted image sample, file name:' + image_name, orig_image)
        pics_index += 1


if __name__ == '__main__':
    main()
