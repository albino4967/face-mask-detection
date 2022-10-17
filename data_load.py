import gdown

file_destinations = {'FaceMaskDetection':'mask_detection_dataset.zip'}

file_id_dic = {'FaceMaskDetection':'1OjTtJ6I7cUtOuEBGtpccKjlWjCxiF9Bu'}

def download_file_from_google_drive(id_, destination):
    url = f'https://drive.google.com/uc?id={id_}'
    output = destination
    gdown.download(url, output, quiet=False)
    print(f'{output} download complete!')

def main():
    download_file_from_google_drive(id_=file_id_dic['FaceMaskDetection'], destination=file_destinations['FaceMaskDetection'])

if __name__ == "__main__" :
    main()