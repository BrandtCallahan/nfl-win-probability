# NFL Win Probability

### How to run:
The nfl_singlegame.py file holds the run_model() function that will allow you to run the model and output a win probability graph like the one shown below. The inputs that that function needs are the season the game is from, the day of the game(s), the number of seasons of data you want the model to train on, the matchup(s) in a list format and a True/False value for whether or not you want the visual displayed. 

To ouput this Colts vs. Titans game:

run_model(<br>
    &emsp;season=2025,<br>
    &emsp;today=pd.to_datetime('2025-09-21').strftime('%Y-%m-%d'),<br>
    &emsp;num_season=10,<br>
    &emsp;matchup=[['IND', 'TEN']],<br>
    &emsp;visualize=True,<br>
)

<img width="640" height="480" alt="ind_v_ten_0921" src="https://github.com/user-attachments/assets/fc170928-ca1e-4894-8718-4ac0d511f6a5" />
