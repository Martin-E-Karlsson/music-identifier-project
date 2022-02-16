from predictions import *


def genre_year():
    file_path = input('Please Input FilePath To The Song: ')
    predict_genre_and_year(file_path)
    return

def genre():
    file_path = input('Please Input FilePath To The Song: ')
    predict_genre(file_path)
    return



def year():
    print('************************************************************')
    selection = input(' Blues: 0 \n Country: 1 \n Disco: 2 \n Hiphop: 3 \n Jazz:4 \n Metal:5 \n Pop: 6 \n Reggae: 7 \n Rock: 8 \n Choose By The Number Which Genre The Song Has That You Want To See The Year In:')
    file_path = input(' Please Input FilePath To The Song: ')
    running = True

    while running:

        if selection == '0':
            result = blues(file_path)
            print(f'Blues from: {result}')
            running = False
        elif selection == '1':
            result = country(file_path)
            print(f'Country from: {result}')
            running = False
        elif selection == '2':
            result = disco(file_path)
            print(f'Disco from: {result}')
            running = False
        elif selection == '3':
            result = hiphop(file_path)
            print(f'Hiphop from: {result}')
            running = False
        elif selection == '4':
            result = jazz(file_path)
            print(f'Jazz from: {result}')
            running = False
        elif selection == '5':
            result = metal(file_path)
            print(f'Metal from: {result}')
            running = False
        elif selection == '6':
            result = pop(file_path)
            print(f'Pop from: {result}')
            running = False
        elif selection == '7':
            result = reggae(file_path)
            print(f'Reggae from: {result}')
            running = False
        elif selection == '8':
            result = rock(file_path)
            print(f'Rock from: {result}')
            running = False

    return


def app():
    menu_choice = input(' Predict Genre And Year: 1 \n Predict Genre: 2 \n Predict Year: 3 \n Choose By The Number What You Want To Do:')
    running = True
    while running:

        if menu_choice == '1':
            genre_year()
            running = False
        elif menu_choice == '2':
            genre()
            running = False
        elif menu_choice == '3':
            year()
            running = False
    return

if __name__ == "__main__":
    app()



# data/country/1950s/1950s4.mp3
# data/hiphop/1990s/1990s4.mp3

# data/hiphop/1990s/1990s4.mp3



