import torch as tr, cv2, numpy as np
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")

class CNN(tr.nn.Module):
    def __init__(self):
        super(CNN, self).__init__() #3x64x64
        self.pool = tr.nn.MaxPool2d(2, 2)

        self.conv1 = tr.nn.Conv2d(3, 64, 5) #60 (in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0)
        #pool 30
        self.conv2 = tr.nn.Conv2d(64, 128,3) #28
        #pool 14
        self.conv3 = tr.nn.Conv2d(128, 256, 3) #12
        #pool 6
        self.conv4 = tr.nn.Conv2d(256, 256, 3) #4
        #pool 2
        self.fc1 = tr.nn.Linear(256 *2 *2, 2048, True) #
        self.fc2 = tr.nn.Linear(2048, 1024, True)
        self.fc3 = tr.nn.Linear(1024, 3, True)

    def forward(self, X): #3*32*32 #3*227*227
        X = tr.nn.functional.relu(self.conv1(X))
        X = self.pool(X)
        X = tr.nn.functional.relu(self.conv2(X))
        X = self.pool(X)
        X = tr.nn.functional.relu(self.conv3(X))
        X = self.pool(X)
        X = tr.nn.functional.relu(self.conv4(X))
        X = self.pool(X)

        X = X.view(X.size(0), -1)

        X = tr.tanh(self.fc1(X))
        X = tr.tanh(self.fc2(X))
        X = tr.nn.functional.softmax(self.fc3(X), dim=1)
        return X

net= CNN()
net.load_state_dict(tr.load('sign_classi_param'))

def predict(img, new_size=64):
    img = cv2.resize(img, (new_size, new_size))
    img = np.array(img, dtype= np.float32) / 255.

    img= img.reshape(1,new_size,new_size,3).transpose((0,3,1,2))

    with tr.no_grad():
        img = tr.from_numpy(img)
        output= net(img)
        output= tr.argmax(output)

    return int(output) #0= not turn; 1= turn right, 2= turn left

# with tr.no_grad():
#     while True:
#         dir= raw_input("file directory: ")
#         if dir == '': break
#
#         img= cv2.imread(dir)
#         cv2.imshow(str(predict(img, 64)), cv2.resize(img, (150,150)))
#         cv2.waitKey(0)
