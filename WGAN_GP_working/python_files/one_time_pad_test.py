from PIL import Image
import  cryptoSystem as cs


# 1 6 4 3


if __name__ == '__main__':
    original_image = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch2\\plain_image.png")
    encrypted_image = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch2\\encrypted_image.png")
    key1 = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch1\\key_iamge.png")
    key3 = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch6\\key_iamge.png")
    key4 = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch4\\key_iamge.png")
    key5 = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch3\\key_iamge.png")
    key6 = Image.open("K:\\MIT_Learnings\\Sem6\\CIP\\final_test\\white_Noise_test\\ch7\\key_iamge.png")

    original_image.show()
    encrypted_image.show()

    dec1 = cs.decrypt(encrypted_image,key1)
    dec1.show()
    dec3 = cs.decrypt(encrypted_image,key3)
    dec3.show()
    dec2 = cs.decrypt(encrypted_image,key4)
    dec2.show()
    dec5 = cs.decrypt(encrypted_image,key5)
    dec5.show()
    dec6 = cs.decrypt(encrypted_image,key6)
    dec6.show()
    


