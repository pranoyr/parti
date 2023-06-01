from parti import reconstruction
if __name__ == '__main__':
    img_path = '/home/pranoy/Downloads/download.jpeg'
    img = reconstruction(img_path=img_path, checkpoint_path="output/models/vit_vq_step_270000.pt")
    img.save('reconstruct.png')