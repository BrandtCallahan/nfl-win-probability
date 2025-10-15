# NFL Win Probability

#### 1. File/Folder Setup
To set things up you and allow the code to run through appropriate data pulls from presaved .csv files, you will need to clone this repository in your computer under this file directory: <br>
f"C:/Users/{os.getlogin()}/personal-github/"

#### 2. Running the Code
The nfl_singlegame.py file holds the run_model() function that will allow you to run the model and output a win probability graph like the one shown below. The inputs that that function needs are the season the game is from, the day of the game(s), the number of seasons of data you want the model to train on, the matchup(s) in a list format and a True/False value for whether or not you want the visual displayed. 

To ouput this Patriots vs. Titans game:

run_model(<br>
    &emsp;season=2025,<br>
    &emsp;today=pd.to_datetime('2025-10-19').strftime('%Y-%m-%d'),<br>
    &emsp;num_season=10,<br>
    &emsp;matchup=[['NWE', 'TEN']],<br>
    &emsp;visualize=True,<br>
)

<img width="640" height="480" alt="nwe_v_ten_1019" src="https://github.com/user-attachments/assets/940c4207-3de1-4819-9d2c-7a27dc0cb2b8" />
