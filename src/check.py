import os.path as osp

import cv2
import matplotlib.cm as cm
import numpy as np
import torch.hub
import os
import model
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchsummary import summary
from grad_cam import BackPropagation, GradCAM,GuidedBackPropagation

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
shape = (24,24)
classes = [
    'Close',
    'Open',
]

def preprocess(image_path):
    transform_test = transforms.Compose([
        transforms.ToTensor()
    ])
    image = cv2.imread(image_path['path'])
    faces = face_cascade.detectMultiScale(
        image,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(1, 1),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        print('no face found')
        face = cv2.resize(image, shape)
        return None, face
    else:
        (x, y, w, h) = faces[0]
        face = image[y:y + h, x:x + w]
        (x, y, w, h) = image_path['left']
        left_eye = face[y:y + h, x:x + w]
        left_eye = cv2.resize(left_eye, shape)
        (x, y, w, h) = image_path['right']
        right_eye = face[y:y + h, x:x + w]
        right_eye = cv2.resize(right_eye, shape)
        return transform_test(Image.fromarray(left_eye).convert('L')), \
               transform_test(Image.fromarray(right_eye).convert('L')), \
               left_eye, right_eye, cv2.resize(face, (48,48))


def get_gradient_image(gradient):
    gradient = gradient.cpu().numpy().transpose(1, 2, 0)
    gradient -= gradient.min()
    gradient /= gradient.max()
    gradient *= 255.0
    return np.uint8(gradient)


def get_gradcam_image(gcam, raw_image, paper_cmap=False):
    gcam = gcam.cpu().numpy()
    cmap = cm.jet_r(gcam)[..., :3] * 255.0
    if paper_cmap:
        alpha = gcam[..., None]
        gcam = alpha * cmap + (1 - alpha) * raw_image
    else:
        gcam = (cmap.astype(np.float) + raw_image.astype(np.float)) / 2
    return np.uint8(gcam)

def guided_backprop_eye(image, name, net):
    img = torch.stack([image[name]])
    bp = BackPropagation(model=net)
    probs, ids = bp.forward(img)
    gcam = GradCAM(model=net)
    _ = gcam.forward(img)

    gbp = GuidedBackPropagation(model=net)
    _ = gbp.forward(img)

    # Guided Backpropagation
    actual_status = ids[:, 0]
    gbp.backward(ids=actual_status.reshape(1, 1))
    gradients = gbp.generate()

    # Grad-CAM
    gcam.backward(ids=actual_status.reshape(1, 1))
    regions = gcam.generate(target_layer='last_conv')

    # Get Images
    prob = probs.data[:, 0]
    if actual_status == 0:
        prob = probs.data[:,1]

    prob_image = np.zeros((shape[0], 60, 3), np.uint8)
    cv2.putText(prob_image, '%.1f%%' % (prob * 100), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                (255, 255, 255), 1, cv2.LINE_AA)

    guided_bpg_image = get_gradient_image(gradients[0])
    guided_bpg_image = cv2.merge((guided_bpg_image, guided_bpg_image, guided_bpg_image))

    grad_cam_image = get_gradcam_image(gcam=regions[0, 0], raw_image=image[name + '_raw'])
    guided_gradcam_image = get_gradient_image(torch.mul(regions, gradients)[0])
    guided_gradcam_image = cv2.merge((guided_gradcam_image, guided_gradcam_image, guided_gradcam_image))
    print(image['path'],classes[actual_status.data], probs.data[:,0] * 100)

    return cv2.hconcat(
        [image[name + '_raw'], prob_image, guided_bpg_image, grad_cam_image, guided_gradcam_image])


def guided_backprop(images, model_name):

    for i, image in enumerate(images):
        left_eye,right_eye, left_eye_raw , right_eye_raw, face = preprocess(image)
        image['left'] = left_eye
        image['right'] = right_eye
        image['left_raw'] = left_eye_raw
        image['right_raw'] = right_eye_raw
        image['face'] = face

    net = model.Model(num_classes=len(classes))
    checkpoint = torch.load(os.path.join('../trained', model_name), map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    summary(net, (1, shape[0], shape[1]))

    result_images = []
    for index, image in enumerate(images):
        left = guided_backprop_eye(image, 'left', net)
        right = guided_backprop_eye(image, 'right', net)
        eyes = cv2.vconcat([left,right])
        image = cv2.hconcat([image['face'],eyes])
        result_images.append(image)

    cv2.imwrite('../test/guided_gradcam.jpg',cv2.resize(cv2.vconcat(result_images), None, fx=2,fy=2))

def main():
    guided_backprop(
        images=[
            {'path': '../test/open_man.jpg', 'right': (17,26,21,21), 'left': (47,24,22,22)},
            {'path': '../test/closed_man.jpg', 'right': (27,43,32,32), 'left': (92,38,43,43)},
            {'path': '../test/open_woman.jpg', 'right': (50,82,77,77), 'left': (184,65,78,78)},
            {'path': '../test/closed_woman.jpg','right': (28,30,46,46), 'left': (83,34,42,42)},
        ],
        model_name='model_11_96_0.1256.t7'
    )


if __name__ == "__main__":
    main()
