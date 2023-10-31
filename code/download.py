import gdown

# # download data (file)
# url = "https://drive.google.com/uc?id=1Gcyc5fuz5869O8cp_xVOHCHeZQgVHTAc"
# output = "data.zip"
# gdown.download(url, output, quiet=False)


# download data
url = "https://drive.google.com/drive/folders/1cp_TCxIZS1hxzr4_2XzdYSpJWzW0Eqrg"
gdown.download_folder(url, quiet=True, use_cookies=False)

# download data
url = "https://drive.google.com/drive/folders/1I0Inj1HMUVGbi_S7oqITFZCyR7PKwXec"
gdown.download_folder(url, quiet=True, use_cookies=False)