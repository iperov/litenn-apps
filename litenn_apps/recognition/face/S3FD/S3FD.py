import operator
from pathlib import Path

import cv2
import litenn as nn
import litenn_apps as lna
import numpy as np

def S3FD_show_test():       
    s3fd = lna.recognition.face.S3FD()

    img = cv2.imread ( lna.test_data.photo.multiple_faces )

    faces = s3fd.extract(img)

    for l,t,r,b,c in faces:
        cv2.rectangle(img, (l,t), (r,b), (0,255,0) )
        
    cv2.imshow("test", img)
    cv2.waitKey(0)
    cv2.destroyWindow("test")
    
class S3FD:
    """
    Face detector
    """
    model_path = Path(__file__).parent / "S3FD.npy"

    def __init__(self):
        S3FD.download_files()

        self.module = S3FDModule()
        self.module.load(S3FD.model_path)


    def extract (self, input_image, is_bgr=True, is_remove_intersects=False, min_face_size=40):
        """
        Extract faces rects

        arguments

         input_image    np.ndarray of shape
                        (HEIGHT, WIDTH)
                        (HEIGHT, WIDTH, 1)
                        (HEIGHT, WIDTH, 3)

         is_bgr(True)   channels format is BGR(opencv)

         is_remove_intersects(False)
                        remove intersecting faces

         min_face_size(40) minimum face size in pixels

        returns

         list of ints (l,t,r,b,c) sorted by maximum area first
         l - left, t - top, r - right, b - bottom, c - percent

        """

        shape_len = len(input_image.shape)
        if shape_len < 2 or shape_len > 3:
            raise ValueError(f'Wrong shape {input_image.shape}')
        if shape_len == 2:
            input_image = input_image[...,None]

        H,W,C = input_image.shape

        if C == 1:
            np.repeat(input_image, 3, -1)
            H,W,C = input_image.shape

        if C != 3:
            raise ValueError(f'Wrong shape {input_image.shape}')

        if is_bgr:
            input_image = input_image[...,::-1]
            is_bgr = False

        WHmax = max(W, H)
        scale_to = 640 if WHmax >= 1280 else WHmax / 2
        scale_to = max(64, scale_to)

        input_scale = WHmax / scale_to

        resize_w = int(W / input_scale)
        resize_h = int(H / input_scale)

        input_image = cv2.resize (input_image, (resize_w, resize_h), interpolation=cv2.INTER_LINEAR)

        input_image = input_image.transpose( (2,0,1) )

        olist = self.module ( nn.Tensor_from_value( input_image[None,...]) )
        
        olist = [x.np() for x in olist ]
        
        
        
        detected_faces = []
        for ltrbc in self.refine (olist):
            l,t,r,b,c = ltrbc
            l,t,r,b = (x*input_scale for x in (l,t,r,b))
            
            bt = b-t
            if min(r-l,bt) < min_face_size:
                continue
            b += bt*0.1
            detected_faces.append ( (int(l),int(t),int(r),int(b), int(c*100),) )

        
        #sort by largest area first
        detected_faces = [ [(l,t,r,b,c), (r-l)*(b-t) ]  for (l,t,r,b,c) in detected_faces ]
        detected_faces = sorted(detected_faces, key=operator.itemgetter(1), reverse=True )
        detected_faces = [ x[0] for x in detected_faces]

        if is_remove_intersects:
            for i in range( len(detected_faces)-1, 0, -1):
                l1,t1,r1,b1,_ = detected_faces[i]
                l0,t0,r0,b0,_ = detected_faces[i-1]

                dx = min(r0, r1) - max(l0, l1)
                dy = min(b0, b1) - max(t0, t1)
                if (dx>=0) and (dy>=0):
                    detected_faces.pop(i)


        return detected_faces
    
        
    def refine(self, olist):
        bboxlist = []

        for i in range(len(olist) // 2):
            ocls, oreg = olist[i * 2], olist[i * 2 + 1]
            stride = 2**(i + 2)    # 4,8,16,32,64,128
            poss = [*zip(*np.where(ocls[:, 1, :, :] > 0.05))]

            #import code
            #code.interact(local=dict(globals(), **locals()))

            for Iindex, hindex, windex in poss:
                axc, ayc = stride / 2 + windex * stride, stride / 2 + hindex * stride
                score = ocls[0, 1, hindex, windex]
                loc = np.ascontiguousarray(oreg[0, :, hindex, windex]).reshape((1, 4))
                priors = np.array([[axc / 1.0, ayc / 1.0, stride * 4 / 1.0, stride * 4 / 1.0]])
                variances = [0.1, 0.2]
                box = self.decode(loc, priors, variances)
                x1, y1, x2, y2 = box[0] * 1.0
                bboxlist.append([x1, y1, x2, y2, score])
        bboxlist = np.array(bboxlist)
        if 0 == len(bboxlist):
            bboxlist = np.zeros((1, 5))

        bboxlist = bboxlist[ self.nms(bboxlist, 0.3) , :]
        bboxlist = [x for x in bboxlist if x[-1] >= 0.5]
        
        return bboxlist
        
    def decode(self, loc, priors, variances):
        boxes = np.concatenate((priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])),
                               1)
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes
        
    def nms(self, dets, thresh):
        """ Perform Non-Maximum Suppression """
        keep = list()
        if len(dets) == 0:
            return keep

        x_1, y_1, x_2, y_2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
        areas = (x_2 - x_1 + 1) * (y_2 - y_1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx_1, yy_1 = np.maximum(x_1[i], x_1[order[1:]]), np.maximum(y_1[i], y_1[order[1:]])
            xx_2, yy_2 = np.minimum(x_2[i], x_2[order[1:]]), np.minimum(y_2[i], y_2[order[1:]])

            width, height = np.maximum(0.0, xx_2 - xx_1 + 1), np.maximum(0.0, yy_2 - yy_1 + 1)
            ovr = width * height / (areas[i] + areas[order[1:]] - width * height)

            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep


    @staticmethod
    def download_files():
        model_path = S3FD.model_path
        model_url = r'https://github.com/iperov/litenn-apps/releases/download/S3FD/S3FD.npy'
        model_sha256 = '68bf15e7779defa5781ac5670f64d5bfc2a98abf3ad6e9b7c13dbe461807ca4c'
        if not model_path.exists() or lna.core.sha256(model_path) != model_sha256:
            lna.core.download_file(model_url, model_path)
            if lna.core.sha256(model_path) != model_sha256:
                raise Exception(f'sha256 of {model_path} is incorrect.')



class S3FDModule(nn.Module):
    def __init__(self):
        self.conv1_1 = nn.Conv2D(3, 64, 3)
        self.conv1_2 = nn.Conv2D(64, 64, 3)

        self.conv2_1 = nn.Conv2D(64, 128, 3)
        self.conv2_2 = nn.Conv2D(128, 128, 3)

        self.conv3_1 = nn.Conv2D(128, 256, 3)
        self.conv3_2 = nn.Conv2D(256, 256, 3)
        self.conv3_3 = nn.Conv2D(256, 256, 3)

        self.conv4_1 = nn.Conv2D(256, 512, 3)
        self.conv4_2 = nn.Conv2D(512, 512, 3)
        self.conv4_3 = nn.Conv2D(512, 512, 3)

        self.conv5_1 = nn.Conv2D(512, 512, 3)
        self.conv5_2 = nn.Conv2D(512, 512, 3)
        self.conv5_3 = nn.Conv2D(512, 512, 3)

        self.fc6 = nn.Conv2D(512, 1024, 3, padding=3)
        self.fc7 = nn.Conv2D(1024, 1024, 1)

        self.conv6_1 = nn.Conv2D(1024, 256, 1)
        self.conv6_2 = nn.Conv2D(256, 512, 3, stride=2)

        self.conv7_1 = nn.Conv2D(512, 128, 1)
        self.conv7_2 = nn.Conv2D(128, 256, 3, stride=2)

        self.conv3_3_norm = L2Norm(256, scale=10)
        self.conv4_3_norm = L2Norm(512, scale=8)
        self.conv5_3_norm = L2Norm(512, scale=5)

        self.conv3_3_norm_mbox_conf = nn.Conv2D(256, 4, 3)
        self.conv3_3_norm_mbox_loc  = nn.Conv2D(256, 4, 3)
        self.conv4_3_norm_mbox_conf = nn.Conv2D(512, 2, 3)
        self.conv4_3_norm_mbox_loc  = nn.Conv2D(512, 4, 3)
        self.conv5_3_norm_mbox_conf = nn.Conv2D(512, 2, 3)
        self.conv5_3_norm_mbox_loc  = nn.Conv2D(512, 4, 3)

        self.fc7_mbox_conf     = nn.Conv2D(1024, 2, 3)
        self.fc7_mbox_loc      = nn.Conv2D(1024, 4, 3)
        self.conv6_2_mbox_conf = nn.Conv2D(512, 2,  3)
        self.conv6_2_mbox_loc  = nn.Conv2D(512, 4,  3)
        self.conv7_2_mbox_conf = nn.Conv2D(256, 2,  3)
        self.conv7_2_mbox_loc  = nn.Conv2D(256, 4,  3)

    def forward(self, x):

        h = nn.relu(self.conv1_1(x))
        h = nn.relu(self.conv1_2(h))
        h = nn.max_pool2D(h)


        h = nn.relu(self.conv2_1(h))
        h = nn.relu(self.conv2_2(h))
        h = nn.max_pool2D(h)


        h = nn.relu(self.conv3_1(h))
        h = nn.relu(self.conv3_2(h))
        h = nn.relu(self.conv3_3(h))
        f3_3 = h
        h = nn.max_pool2D(h)

        h = nn.relu(self.conv4_1(h))
        h = nn.relu(self.conv4_2(h))
        h = nn.relu(self.conv4_3(h))
        f4_3 = h
        h = nn.max_pool2D(h)

        h = nn.relu(self.conv5_1(h))
        h = nn.relu(self.conv5_2(h))
        h = nn.relu(self.conv5_3(h))
        f5_3 = h
        h = nn.max_pool2D(h)

        h = nn.relu(self.fc6(h))
        h = nn.relu(self.fc7(h))
        ffc7 = h

        h = nn.relu(self.conv6_1(h))
        h = nn.relu(self.conv6_2(h))

        f6_2 = h

        h = nn.relu(self.conv7_1(h))
        h = nn.relu(self.conv7_2(h))
        f7_2 = h

        f3_3 = self.conv3_3_norm(f3_3)
        f4_3 = self.conv4_3_norm(f4_3)
        f5_3 = self.conv5_3_norm(f5_3)


        cls1 = self.conv3_3_norm_mbox_conf(f3_3)
        reg1 = self.conv3_3_norm_mbox_loc(f3_3)
        cls2 = nn.softmax(self.conv4_3_norm_mbox_conf(f4_3), 1)
        reg2 = self.conv4_3_norm_mbox_loc(f4_3)
        cls3 = nn.softmax(self.conv5_3_norm_mbox_conf(f5_3), 1)
        reg3 = self.conv5_3_norm_mbox_loc(f5_3)
        cls4 = nn.softmax(self.fc7_mbox_conf(ffc7), 1)
        reg4 = self.fc7_mbox_loc(ffc7)
        cls5 = nn.softmax(self.conv6_2_mbox_conf(f6_2), 1)
        reg5 = self.conv6_2_mbox_loc(f6_2)
        cls6 = nn.softmax(self.conv7_2_mbox_conf(f7_2), 1)
        reg6 = self.conv7_2_mbox_loc(f7_2)

        cls1 = nn.concat( [ nn.reduce_max( cls1[:,0:3,:,:], 1, keepdims=True ),cls1[:,3:4,:,:] ], 1 )
        cls1 = nn.softmax(cls1, 1)

        return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale=1.0):
        self.n_channels = n_channels
        self.weight = nn.Tensor( (n_channels,), init=nn.initializer.Scalar(scale) )
        super().__init__(saveables=['weight'])

    def forward(self, x):
        x = x / (nn.sqrt( nn.reduce_sum( nn.square(x), axes=1, keepdims=True ) ) + 1e-10) * self.weight.reshape( (1,-1,1,1) )
        return x
