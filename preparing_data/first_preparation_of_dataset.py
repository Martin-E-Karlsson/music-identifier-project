import os


genres = ['1970s', '1980s', '1990s', '2000s'] # Dessa byts ut beroende på vilka årtal du har i datasetet.

def prepare_data(file_map, g):
    os.chdir(f"D:/pythonProject1/music_ai_with_new_pycharm/music_identifier/music-identifier-project/data/{file_map}/{g}")
    for count, f in enumerate(os.listdir()):
        f_name, f_ext = os.path.splitext(f)
        f_name = f"{g}" + str(count)

        new_name = f'{f_name}{f_ext}'
        os.rename(f, new_name)
    return

for g in genres:
    run = prepare_data(file_map='metal',g=g) # file_map byts ut beroende på vilken genre du vill köra.


