# Projektbeskrivning


 Projektbeskrivning:

    Vi tänker skapa en ai-modell som kan datera musik till ett årtionde eller årtal. 
    Kan musik kopplas till en specifik tidsperiod? Om så är fallet, vilka låtar passar in i den period då de skapats och vilka är svårare att placera? 
    Kan musikaliska trender från en tidsperiod läsas av som mönster av en ai-modell. 
    Det är frågorna vi vill besvara. Vi kommer troligtvis använda oss av ett neurala nätverk för att träna en modell men är öppna för att se vilka svar vi skulle kunna få utav andra typer modeller. 
    Förutom att hantera matriser och tensorer så vet jag inte hur nära vi kommer algoritmerna som de verktyg vi använder i träningen av våra modeller. 
    Tjänster som vill identifiera och kategorisera musik, eller rekommendera musik till användare borde vara intresserade. Alternativt skulle modellen kunna byggas in i en app för folk som för nöje eller kunskap veta när en låt skapats.


 Teori:

    Vi har valt att använda oss av olika modeller av sekventiella Neurala Nätverk. Vi har valt att använda oss av Tensorflow som verktyg för att bygga dessa neurala nätverk.
    Dom modeller vi valde att använda oss av är:

    -	Recurrent neural network: Med 4 lager.
    -	Convolutional neural network: Med 13 lager.
    -	Multilayer:  Med 5 lager var av det första lagret är ett Flatten lager.
    -	Multilayer overfitting: Med 8 lager, var av första lagret är ett Flatten lager, och lager 3, 5, 7 är Dropout lager.

    Vi kom fram till att Multilayer_overfitting modellen fungerade bäst, grafer/statistik på detta kan ses i mappen models_history.
    Problem och lösningar som uppstått under projektet:
    Från början hade vi lagt alla låtar och alla årtal i samma dataset. Detta blev väldigt svårt för det neurala nätverket få ut någonting av. Så lösningen på detta blev att vi fick bygga om datasetet till flera dataset. 
    Så strukturen på dataseten blev istället, ett dataset med alla genrer, och sedan har varje genrer ett dataset där årtalet är utkomsterna. Så vår app består av flera tränade modeller.

     
 Strukturen för hur vår app körs:

    1. Den predictar genrer på låten den får in.
    2. Sedan beroende på vilken genrer den predictar. Så går den till den tränade genrer modell och predictar vilket årtal låten är ifrån.    


# Running

 Preparing data:

    1. conversion_utils.py : If you want to convert mp3 song to wav.
    2. first_preparation_of_dataset.py : Changes name and add a number to every song.
    3. prepare_dataset.py : Mapping and pick out mfcc values to a json dataset.

 Model training:
    
    1. We have four different neural networks you can train with, multilayer_model_overfitting.py is the one that works best and is also the one on which the app runs.
       You can find all neural networks in neural_networks/

    2. Run multilayer_model_overfitting.py on all files in data_json/

 App:

    Run the app by running main.py

    The app is built with the help of 10 different models, the first model predict what genre it is on the song. 
    Then depending on what genre it predict it is. 
    Then it goes on to predict what year the song is from, in the specific genre.



