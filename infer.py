from .crop_video import process_video
from .demo import demo
import os
from facenet_pytorch import MTCNN, extract_face
# import cv2
from PIL import Image, ImageDraw 



class Animation:
    def __init__(self):
        self.image_shape= (256, 256)
        self.increase= 0.1
        self.iou_with_initial= 0.25
        # self.inp= 'image_animation/video/id10784#CDDXIPQGNF4#000840#000980.mp4'
        self.min_frames= 150
        self.cpu= False

        self.config = 'image_animation/config/vox-adv-256.yaml' ##
        self.checkpoint = 'image_animation/vox-adv-cpk.pth.tar'
        self.driving_video ='crop.mp4'
        self.relative= True
        self.adapt_scale = True
        self.find_best_frame = False
        self.best_frame = None

    def _crop_video(self, inp):
        self.inp = inp
        commands = process_video(self)
        for command in commands:
            print (command)
            os.system(command)

    def animate(self, source_video, source_image='image_animation/image/einstein.png', 
                result_video = 'result.mp4'):
        self._crop_video(source_video)
        self.source_image = source_image
        self.result_video = result_video
        demo(self)


class FaceDetect:
    def __init__(self, thresholds = [0.9, 0.9, 0.9], min_face_size = 100):
        self.mtcnn = MTCNN(thresholds=thresholds, select_largest=True, post_process=False, device='cuda:0', min_face_size=min_face_size)
    
    def detect(self,img_ls, crop_size = None, mode = 'Detect_bool', save_faces = False, save_annotate = False):
        """face detection

        Args:
            img_ls (list): list of array
            crop_size (tuple, optional): crop images with (left, top, right, bottom). Defaults to None.
            mode (str, optional): There're 3 modes, 'Detect', 'Detect_bool', and 'Extract'. 
                                    If you only want to know whether there're any faces, use 'Detect_bool' mode. 
                                    If you want to get boxes and probs of faces, use 'Detect'.
                                    If you want to get all information about faces, use 'Extract'.
                                    Defaults to 'Detect_bool'.
            face_num (int, optional): Number of faces to be extracted. Defaults to 1.
            save_faces (bool, optional): For 'Extract' mode. Defaults to False.
            save_annotate (bool, optional): For 'Extract' mode. Save images with annotations. Defaults to False.

        Returns:
            tuple: depends on the mode.

        """
        if crop_size:
            for i, img in enumerate(img_ls):
                img_ls[i] = img.crop(crop_size)

        try:
            boxes, probs = self.mtcnn.detect(img_ls)
        except Exception as e:
            print(f'{e} \n...add crop_size=(left, top, right, bottom) to make images the same')

        if mode == 'Detect_bool':
            return isinstance(boxes, np.ndarray)
        elif mode == 'Detect':
            return boxes, probs 
        elif mode == 'Extract':
            faces = []
            annotates = []
            for id_, img in enumerate(img_ls):
                face_batch = []
                img_annotate = img.copy()
                draw = ImageDraw.Draw(img_annotate)
                for i, box in enumerate(boxes[id_]):
                    box_head = [box[0]-box[0]/8, box[1]-box[1]/5, box[2]+box[2]/8, box[3]+box[3]/10]
                    boxes[id_][i] = np.asarray(box_head)
                    draw.rectangle(box_head, width=5)  # box.tolist()
                    if save_faces:
                        face_batch.append(extract_face(img, box_head, save_path=os.path.join(path,f'detected_face_{id_}-{i}.png')))
                    else:
                        face_batch.append(extract_face(img, box_head))
                faces.append(face_batch)
                annotates.append(np.asarray(img_annotate))
                if save_annotate:
                    img_annotate.save(os.path.join(path,f'annotated_faces_{id_}.png'))
            return boxes, probs, annotates, faces
        else:
            print(f"Error: there's no mode called {mode}")



if __name__ == "__main__":
    a = Animation()
    a.crop_video()
    a.animate()



    # import images
    path = 'data_faces_detect'
    files = os.listdir(path)
    images = []
    for f in files:    
        print(f)
        img = Image.open(os.path.join(path,f))
        images.append(img)


# have_face = detect(images, crop_size = (0,0,512,341))  
# boxes, probs = detect(images,mode='Detect', crop_size = (0,0,512,341)) 
# boxes, probs, annotates, faces = detect(images, mode='Extract', crop_size = (0,0,512,341)) #, face_extract = True, save_faces = True, save_annotate = True)
# detect(images_2, mode='Extract',  save_faces = True, save_annotate = True)


