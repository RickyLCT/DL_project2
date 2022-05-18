import os
from PIL import Image, ImageDraw, ImageFont

characters = ['0','1','2','3','4','5','6','7','8','9',
        'a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z',
        'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

font_path = os.path.abspath('./dataset/font/fonts')
dataset_path = os.path.abspath("./dataset/character_dataset")

def getImageName(font_path, char, idx):
    font_file = os.path.basename(font_path)
    font_name, _ = os.path.splitext(font_file)
    return f"{font_name}_{char}_{idx}.png"

def createImage(font_path, char, idx, output_path):
    font_size = 45
    try:
        img = Image.new("L", (64, 64), "white")
        fnt = ImageFont.truetype(font_path, font_size)
        d = ImageDraw.Draw(img)
        d.text((32, 32), char, fill="black", anchor="mm", font=fnt)
        img.save(os.path.join(output_path, getImageName(font_path, char, idx)))
    except:
        print("error occured, continue")
        
        
def generateDataset(dataset_path, font_path):
    # create file if not exist
    if not os.path.exists(dataset_path):
        print(f"{dataset_path} not exists, create it")
        os.mkdir(dataset_path)
        
    for font in os.listdir(font_path):
        font_dir = os.path.join(font_path, font)
        if os.path.isdir(font_dir):
            for font_file in os.listdir(font_dir):
                filename, ext = os.path.splitext(font_file)
                if ext == ".ttf" or ext == ".otf":
                    print(f"starting generate {filename}...")
                    font_path = os.path.join(font_dir, font_file)
                for idx, char in enumerate(characters):
                    print(f"generating {char}...")
                    createImage(font_path, char, dataset_path, idx)
                print("finished")
                


    
generateDataset(dataset_path, font_path)