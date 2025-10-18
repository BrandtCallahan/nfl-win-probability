# NFL Win Probability

#### 1. File/Folder Setup
To set things up you and allow the code to run through appropriate data pulls from presaved .csv files, you will need to clone this repository in your computer under this file directory: <br>
f"C:/Users/{os.getlogin()}/personal-github/"

#### 2. Refreshing/Adding Data
In order to add or refresh data, you can run the refresh_data() function which can be found in the nfl_singlegame.py file. The only parameter this function needs is the year you are wanting to refresh/add. So the function will look like this to refresh 2025 data:<br>

refresh_data(2025)

#### 3. Running the Code
The nfl_singlegame.py file also holds the run_model() function that will allow you to run the model and output a win probability graph like the one shown below. The inputs that that function needs are the season the game is from, the day of the game(s), the number of seasons of data you want the model to train on, the matchup(s) in a list format and a True/False value for whether or not you want the visual displayed. If you want to run multiple matchups, then you can just add those matchups to the matchup parameter in a list format. The abbreviations for each team can be found at the bottom (or in the team.csv file).

To ouput this Patriots vs. Titans game:

run_model(<br>
    &emsp;season=2025,<br>
    &emsp;today=pd.to_datetime('2025-10-19').strftime('%Y-%m-%d'),<br>
    &emsp;num_season=10,<br>
    &emsp;matchup=[['NWE', 'TEN']],<br>
    &emsp;visualize=True,<br>
)

<img width="640" height="480" alt="nwe_v_ten_1019" src="https://github.com/user-attachments/assets/b1e489d8-492a-4f42-b33b-e9add51734e3" />

ARI: Arizona Cardinals <br>
ATL: Atlanta Falcons <br>
BAL: Baltimore Ravens <br>
BUF: Buffalo Bills <br>
CAR: Carolina Panthers <br>
CHI: Chicago Bears <br>
CIN: Cincinnati Bengals <br>
CLE: Cleveland Browns <br>
DAL: Dallas Cowboys <br>
DEN: Denver Broncos <br>
DET: Detroit Lions <br>
GNB: Green Bay Packers <br>
HOU: Houston Texans <br>
IND: Indianapolis Colts <br>
JAX: Jacksonville Jaguars <br>
KAN: Kansas City Chiefs <br>
LAC: Los Angeles Chargers <br>
LAD: Los Angeles Rams <br>
LVR: Las Vegas Raiders <br>
MIA: Miami Dolphins <br>
MIN: Minnesota Vikings <br>
NWE: New England Patriots <br>
NOR: New Orleans Saints <br>
NYG: New York Giants <br>
NYJ: New York Jets <br>
PHI: Philadelphia Eagles <br>
PIT: Pittsburgh Steelers <br>
SEA: Seattle Seahawks <br> 
SFO: San Francisco 49ers <br>
TAM: Tampa Bay Buccaneers <br>
TEN: Tennessee Titans <br>
WAS: Washington Commanders <br>
