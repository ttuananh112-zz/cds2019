import torch as tr, cv2, numpy as np
from CNN import CNN
device = tr.device("cuda" if tr.cuda.is_available() else "cpu")
print('done lib')
net= CNN()
net.load_state_dict(tr.load('traffic_sign_param'))

print('done param')

def switch(x):
    return {
        0: 'not',
        1: 'right',
        2: 'left',
    }.get(x, 'nan')

def predict(img, new_size):
    img = cv2.resize(img, (new_size, new_size))
    img = np.array(img, dtype= np.float32) / 255.

    img= img.reshape(1,new_size,new_size,3).transpose((0,3,1,2))

    img= tr.from_numpy(img)

    output= net(img)
    output= tr.argmax(output)
    return int(output)

with tr.no_grad():
    while True:
        dir= raw_input("file directory: ")
        if dir == '': break

    # for i in range (1,9):
    #     dir= 'other imgs/cap'+ str(i) +'.png'

        img= cv2.imread(dir)
        cv2.imshow(switch(predict(img, 64)), cv2.resize(img, (150,150)))
        cv2.waitKey(0)