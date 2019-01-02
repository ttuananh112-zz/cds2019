import torch as tr, cv2, numpy as np
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

class CNN32(tr.nn.Module):
    def __init__(self):
        super(CNN32, self).__init__() #3x32x32
        self.pool = tr.nn.MaxPool2d(2, 2)

        self.conv1 = tr.nn.Conv2d(3, 64, 5) #28 (in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        #pool 14
        self.conv2 = tr.nn.Conv2d(64, 128,3) #12
        #pool 6
        self.conv3 = tr.nn.Conv2d(128, 256, 3) #4
        #pool 2
        self.fc1 = tr.nn.Linear(256 *2 *2, 2048) #
        self.fc2 = tr.nn.Linear(2048, 1024)
        self.fc3 = tr.nn.Linear(1024, 3)


    def forward(self, X):
        X = tr.nn.functional.relu(self.conv1(X))
        X = self.pool(X)
        X = tr.nn.functional.relu(self.conv2(X))
        X = self.pool(X)
        X = tr.nn.functional.relu(self.conv3(X))
        X = self.pool(X)

        X = X.view(X.size(0), -1)

        X = tr.tanh(self.fc1(X))
        X = tr.tanh(self.fc2(X))
        X = tr.nn.functional.softmax(self.fc3(X), dim=1)
        return X

net= CNN32()
net.load_state_dict(tr.load('sign_classi_param32'))

def predict(img, new_size=32):
    img = cv2.resize(img, (new_size, new_size))
    img = np.array(img, dtype= np.float32) / 255.

    img= img.reshape(1,new_size,new_size,3).transpose((0,3,1,2))

    with tr.no_grad():
        img = tr.from_numpy(img)
        output= net(img)
        output= tr.argmax(output)

    return int(output) #0= not turn; 1= turn right, 2= turn left

# with tr.no_grad():
    # while True:
    #     dir= raw_input("file directory: ")
    #     if dir == '': break
#     for i in range(1,27):
#         dir= 'other imgs/o' + str(i) + '.png'

#         img= cv2.imread(dir)
#         cv2.imshow(str(predict(img)), cv2.resize(img, (150,150)))
#         cv2.waitKey(0)
