## Data Dictionaries
```python
import os
import pandas as pd

# define Parquet file directory
RAW_DATA_DIR = "data/raw"

# specify names of raw data files
parquet_files = ["all_players_data_sdk.parquet",
                 "all_teams_data_sdk.parquet",
                 "games_data_seasons_2021_2022_2023_2024_2025_sdk.parquet",
                 "player_game_stats_seasons_2021_2022_2023_2024_2025.parquet"]

# function to load raw Parquet data files as DataFrame objects
def load_parquet_files(file_list, directory):
    """
    Loads multiple Parquet files from a specified directory into pandas DataFrames.

    Args:
        file_list (list): A list of Parquet file names.
        directory (str): The directory where the Parquet files are located.

    Returns:
        dict: A dictionary where keys are the file names (without the .parquet extension)
              and values are the corresponding pandas DataFrames.
              Returns an empty dictionary if no files are loaded or if errors occur.
    """
    dataframes = {}
    for parquet_filename in file_list:
        # define Parquet file path
        file_path = os.path.join(directory, parquet_filename)
        try:
            # load the Parquet file into a DataFrame
            df = pd.read_parquet(file_path)
            # use the filename (without extension) as the key for the dictionary
            df_name = os.path.splitext(parquet_filename)[0]
            dataframes[df_name] = df
            print(f"Successfully loaded '{parquet_filename}' as '{df_name}'")
        except FileNotFoundError:
            print(f"Error: The file '{parquet_filename}' was not found at '{file_path}'")
        except Exception as e:
            print(f"Error loading '{parquet_filename}': {e}")
    return dataframes

# load the Parquet files using the function
loaded_data = load_parquet_files(parquet_files, RAW_DATA_DIR)
```

# Data Dictionary: df_player_stats 

```python
# load player stats data as DataFrame
if "player_game_stats_seasons_2021_2022_2023_2024_2025" in loaded_data:
    df_player_stats = loaded_data["player_game_stats_seasons_2021_2022_2023_2024_2025"]
    print(df_player_stats.columns.tolist())
    print(df_player_stats[['id', 'player_id', 'player_team_id', 'team_id', 'game_home_team_id', 'game_visitor_team_id', 'game_id']].head())
    df_player_stats.head()
```

### This table describes the columns present in the `df_player_stats` DataFrame, which contains player statistics for individual games.
### Specifically, `df_player_stats` contains historical player performance statistics for individual games played between the 2021-2025 (Current) seasons.

| Column Name          | Data Type        | Description                                                                  | Source/Notes                                                                                               |
|----------------------|------------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------|
| **Player Stats Columns**|                  | **(Directly from the 'stats' endpoint for each game performance)** |                                                                                                            |
| `id`                 | Integer          | Unique identifier for the specific statistic entry/record.                   | Main field. API: `stats.id`                                                                                |
| `min`                | String / Integer | Minutes played in the game. (API often returns as string, e.g., "30")        | Main field. API: `stats.min`                                                                               |
| `fgm`                | Integer / Float  | Field goals made.                                                            | Main field. API: `stats.fgm`                                                                               |
| `fga`                | Integer / Float  | Field goals attempted.                                                       | Main field. API: `stats.fga`                                                                               |
| `fg_pct`             | Float            | Field goal percentage (fgm/fga).                                             | Main field. API: `stats.fg_pct`                                                                            |
| `fg3m`               | Integer / Float  | Three-point field goals made.                                                | Main field. API: `stats.fg3m`                                                                              |
| `fg3a`               | Integer / Float  | Three-point field goals attempted.                                           | Main field. API: `stats.fg3a`                                                                              |
| `fg3_pct`            | Float            | Three-point field goal percentage (fg3m/fg3a).                               | Main field. API: `stats.fg3_pct`                                                                           |
| `ftm`                | Integer / Float  | Free throws made.                                                            | Main field. API: `stats.ftm`                                                                               |
| `fta`                | Integer / Float  | Free throws attempted.                                                       | Main field. API: `stats.fta`                                                                               |
| `ft_pct`             | Float            | Free throw percentage (ftm/fta).                                             | Main field. API: `stats.ft_pct`                                                                            |
| `oreb`               | Integer / Float  | Offensive rebounds.                                                          | Main field. API: `stats.oreb`                                                                              |
| `dreb`               | Integer / Float  | Defensive rebounds.                                                          | Main field. API: `stats.dreb`                                                                              |
| `reb`                | Integer / Float  | Total rebounds (oreb + dreb).                                                | Main field. API: `stats.reb`                                                                               |
| `ast`                | Integer / Float  | Assists.                                                                     | Main field. API: `stats.ast`                                                                               |
| `stl`                | Integer / Float  | Steals.                                                                      | Main field. API: `stats.stl`                                                                               |
| `blk`                | Integer / Float  | Blocks.                                                                      | Main field. API: `stats.blk`                                                                               |
| `turnover`           | Integer / Float  | Turnovers.                                                                   | Main field. API: `stats.turnover`                                                                          |
| `pf`                 | Integer / Float  | Personal fouls.                                                              | Main field. API: `stats.pf`                                                                                |
| `pts`                | Integer / Float  | Points scored.                                                               | Main field. API: `stats.pts`                                                                               |
| **Player Columns** |                  | **(Flattened from the nested 'player' object in the API response)** |                                                                                                            |
| `player_id`          | Integer          | Unique identifier for the player.                                            | Nested: `player.id`                                                                                        |
| `player_first_name`  | String           | First name of the player.                                                    | Nested: `player.first_name`                                                                                |
| `player_last_name`   | String           | Last name of the player.                                                     | Nested: `player.last_name`                                                                                 |
| `player_position`    | String           | Position of the player (e.g., 'G', 'F', 'C', 'G-F').                         | Nested: `player.position`                                                                                  |
| `player_team_id`     | Integer          | ID of the team the player is generally associated with (from player object). | Nested: `player.team_id`. This is the `team_id` *within the player object*, distinct from the `team_id` below. |
| **Team Columns** |                  | **(Flattened from the nested 'team' object - team for *this specific game stat*)** |                                                                                                            |
| `team_id`            | Integer          | Unique identifier for the team for which the player recorded these stats.    | Nested: `team.id`. This is the team the player played for *in this game*.                                    |
| `team_abbreviation`  | String           | Abbreviation for the team (e.g., 'LAL', 'BKN').                              | Nested: `team.abbreviation`                                                                                |
| `team_full_name`     | String           | Full name of the team (e.g., 'Los Angeles Lakers').                          | Nested: `team.full_name`                                                                                   |
| **Game Columns** |                  | **(Flattened from the nested 'game' object in the API response)** |                                                                                                            |
| `game_id`            | Integer          | Unique identifier for the game in which these stats were recorded.           | Nested: `game.id`                                                                                          |
| `game_date`          | String (Date)    | Date of the game (YYYY-MM-DD format).                                        | Nested: `game.date`                                                                                        |
| `game_season`        | Integer          | The year the season concludes (e.g., 2022 for the 2021-2022 NBA season).     | Nested: `game.season`                                                                                      |
| `game_home_team_id`  | Integer          | Unique identifier for the home team in this game.                            | Nested: `game.home_team_id`                                                                                |
| `game_visitor_team_id`| Integer          | Unique identifier for the visitor team in this game.                         | Nested: `game.visitor_team_id`                                                                             |
| `game_postseason`    | Boolean          | True if the game is a postseason game, False otherwise.                       | Nested: `game.postseason`                                                                                  |

**Notes on Data Types:**
* **Integer / Float:** Many statistical counts are whole numbers but might be represented as floats if there are missing values (`NaN`, which forces a float dtype in Pandas) or if the API returns them as floats. The Pydantic models often define these as `Optional[float]`.
* **String / Integer for `min`:** The 'min' (minutes played) field is often a string in the API response (e.g., "37:00" or just "37"). Your parsing logic might convert this, or it might remain a string. It's listed as `Optional[str]` in the `NBAStats` Pydantic model.
* **Percentages:** Fields like `fg_pct` are typically floats between 0.0 and 1.0.
* **IDs:** These are unique identifiers and are generally integers.


# Data Dictionary: df_players

```python
# load all players data as DataFrame, access each DataFrame by name
if "all_players_data_sdk" in loaded_data:
    df_players = loaded_data["all_players_data_sdk"]
    print(df_players.columns.tolist())
    df_players.head()
```

### This table describes the columns present in the `df_players` DataFrame, which contains information about individual NBA players.

| Column Name         | Data Type       | Description                                                                    | Source/Notes                                                                 |
|---------------------|-----------------|--------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Player Info** |                 | **(Directly from the 'player' object in the API response)** |                                                                              |
| `id`                | Integer         | Unique identifier for the player.                                              | Main field. API: `player.id`                                                 |
| `first_name`        | String          | First name of the player.                                                      | Main field. API: `player.first_name`                                         |
| `last_name`         | String          | Last name of the player.                                                       | Main field. API: `player.last_name`                                          |
| `position`          | String          | Player's primary position (e.g., 'G', 'F', 'C', 'G-F'). Can be empty/None.     | Main field. API: `player.position`                                           |
| `height`            | String          | Player's height (e.g., "6-6"). Stored as string. Can be empty/None.          | Main field. API: `player.height` (was `height_feet` & `height_inches` in older API versions) |
| `weight`            | String / Integer| Player's weight in pounds. Stored as string if API provides it as such, might be numeric. Can be empty/None. | Main field. API: `player.weight` (was `weight_pounds` in older API versions) |
| `jersey_number`     | String / Integer| Player's jersey number. Can be empty/None.                                   | Main field. API: `player.jersey_number`                                      |
| `college`           | String          | College the player attended. Can be empty/None (e.g., international, HS).    | Main field. API: `player.college`                                            |
| `country`           | String          | Player's country of origin/nationality.                                        | Main field. API: `player.country`                                            |
| `draft_year`        | Integer / Float | Year the player was drafted. Can be NaN/None if undrafted.                     | Main field. API: `player.draft_year`                                         |
| `draft_round`       | Integer / Float | Round in which the player was drafted. Can be NaN/None if undrafted.           | Main field. API: `player.draft_round`                                        |
| `draft_number`      | Integer / Float | Overall pick number in the draft. Can be NaN/None if undrafted.                | Main field. API: `player.draft_number`                                       |
| **Team Info** |                 | **(Flattened from the nested 'team' object associated with the player)** |                                                                              |
| `team_id`           | Integer         | Unique identifier for the player's current/most recent team.                   | Nested: `player.team.id`                                                     |
| `team_abbreviation` | String          | Abbreviation for the player's team (e.g., 'OKC', 'ATL').                       | Nested: `player.team.abbreviation`                                           |
| `team_full_name`    | String          | Full name of the player's team (e.g., 'Oklahoma City Thunder').                | Nested: `player.team.full_name`                                              |
| `team_conference`   | String          | Conference of the player's team (e.g., 'West', 'East').                        | Nested: `player.team.conference`                                             |
| `team_division`     | String          | Division of the player's team (e.g., 'Northwest', 'Southeast').                | Nested: `player.team.division`                                               |
| `team_city`         | String          | City of the player's team.                                                     | Nested: `player.team.city`                                                   |

**Notes on Data Types and Values:**

* **`NaN` / `None`:** As seen in your screenshot for `draft_year`, `draft_round`, `draft_number` for some players (like Jaylen Adams who was likely undrafted), these fields can be missing. Pandas will represent these as `NaN` (Not a Number) for numeric types or `None` (which might also become `NaN` in a mixed-type column) for object/string types.
* **`height` and `weight`**: The API provides height as a string (e.g., "6-4"). Weight is also often a string from the API but might appear numeric if all values are numbers.
* **`position`**: Can sometimes be empty or have combined positions like "F-C".
* **`team_id` (Player vs. Stat context):**
    * In `df_players`, `team_id` (and the other `team_*` columns) refers to the team information *nested within the player object*. This generally represents the player's current or most recently known team affiliation *at the time the player data was fetched*.
    * This is distinct from the `team_id` you'd find in `df_player_stats`, where `team_id` refers to the team for which a specific game statistic was recorded.
* **`player_team_id` vs `team_id` in `df_player_stats`**: In your `df_player_stats` (from the previous data dictionary we discussed), you had both `player_team_id` and `team_id`.
    * `player_team_id` came from `player.team_id` within the stat's player object.
    * `team_id` came from the `team.id` directly associated with the stat (i.e., the team the player played *for* in that game).
    This distinction is important for player movement and trades. In `df_players`, you only have one set of team columns, prefixed with `team_`, representing the team associated with the player record.


# Data Dictionary: df_games

```python
# load games data as DataFrame
if "games_data_seasons_2021_2022_2023_2024_2025_sdk" in loaded_data:
    df_games = loaded_data["games_data_seasons_2021_2022_2023_2024_2025_sdk"]
    print(df_games.columns.tolist())
    df_games.head()
```

### This table describes the columns present in the `df_games` DataFrame, which contains information about individual NBA games.

| Column Name             | Data Type       | Description                                                                  | Source/Notes                                                                 |
|-------------------------|-----------------|------------------------------------------------------------------------------|------------------------------------------------------------------------------|
| **Main Game Fields** |                 | **(Directly from the 'game' object in the API response)** |                                                                              |
| `id`                    | Integer         | Unique identifier for the game.                                              | Main field. API: `game.id`                                                   |
| `date`                  | String (Date)   | Date of the game (YYYY-MM-DD format).                                        | Main field. API: `game.date`                                                 |
| `season`                | Integer         | The year the season concludes (e.g., 2023 for the 2022-2023 NBA season).     | Main field. API: `game.season`                                               |
| `status`                | String          | Status of the game (e.g., "Final", "Scheduled", "Postponed").                | Main field. API: `game.status`                                               |
| `period`                | Integer / Float | Current period of the game (e.g., 4 for final). Can be float if NaNs.       | Main field. API: `game.period`                                               |
| `time`                  | String          | Game time status (e.g., "Final", "Q4 02:30"). Can be empty.                  | Main field. API: `game.time`                                                 |
| `postseason`            | Boolean         | True if the game is a postseason game, False otherwise.                       | Main field. API: `game.postseason`                                           |
| `home_team_score`       | Integer / Float | Score of the home team. Can be NaN if game not played/final.                | Main field. API: `game.home_team_score`                                      |
| `visitor_team_score`    | Integer / Float | Score of the visitor team. Can be NaN if game not played/final.             | Main field. API: `game.visitor_team_score`                                   |
| **Home Team Fields** |                 | **(Flattened from the nested 'home_team' object in the API response)** |                                                                              |
| `home_team_id`          | Integer         | Unique identifier for the home team.                                         | Nested: `game.home_team.id`                                                  |
| `home_team_abbreviation`| String          | Abbreviation for the home team (e.g., 'GSW', 'LAL').                         | Nested: `game.home_team.abbreviation`                                        |
| `home_team_full_name`   | String          | Full name of the home team (e.g., 'Golden State Warriors').                  | Nested: `game.home_team.full_name`                                           |
| `home_team_conference`  | String          | Conference of the home team (e.g., 'West', 'East').                          | Nested: `game.home_team.conference`                                          |
| `home_team_division`    | String          | Division of the home team (e.g., 'Pacific', 'Atlantic').                     | Nested: `game.home_team.division`                                            |
| `home_team_city`        | String          | City of the home team.                                                       | Nested: `game.home_team.city`                                                |
| **Visitor Team Fields** |                 | **(Flattened from the nested 'visitor_team' object in the API response)** |                                                                              |
| `visitor_team_id`       | Integer         | Unique identifier for the visitor team.                                      | Nested: `game.visitor_team.id`                                               |
| `visitor_team_abbreviation`| String       | Abbreviation for the visitor team.                                           | Nested: `game.visitor_team.abbreviation`                                     |
| `visitor_team_full_name`| String          | Full name of the visitor team.                                               | Nested: `game.visitor_team.full_name`                                        |
| `visitor_team_conference`| String         | Conference of the visitor team.                                              | Nested: `game.visitor_team.conference`                                       |
| `visitor_team_division` | String          | Division of the visitor team.                                                | Nested: `game.visitor_team.division`                                         |
| `visitor_team_city`     | String          | City of the visitor team.                                                    | Nested: `game.visitor_team.city`                                             |

**Notes on Data Types and Values:**

* **`NaN` / `None`:** For games that haven't been played or are not yet final, fields like `home_team_score`, `visitor_team_score`, `period`, and `time` might be `NaN` or `None`.
* **`season`:** Typically refers to the year the NBA season *ends*. For example, the 2022-2023 season would be represented as `season: 2023`.
* **`status` vs. `time`:**
    * `status` gives a general state like "Final", "Scheduled", "Halftime".
    * `time` can provide more specific details, like the clock time in a quarter if the game is live, or be empty/ "Final" for completed games.
* **Consistency:** The prefixed columns (`home_team_*`, `visitor_team_*`) are a direct result of your `_parse_and_flatten_data` function processing the `nested_game_fields` you defined.


# Data Dictionary: df_teams

```python
# load teams data as DataFrame
if "all_teams_data_sdk" in loaded_data:
    df_teams = loaded_data["all_teams_data_sdk"]
    df_teams.head()
```

### This table describes the columns present in the `df_teams` DataFrame, which contains information about individual NBA teams. All fields are directly sourced from the main attributes of each team object returned by the API.

| Column Name  | Data Type | Description                                                     | Source/Notes                       |
|--------------|-----------|-----------------------------------------------------------------|------------------------------------|
| `id`         | Integer   | Unique identifier for the team.                                 | Main field. API: `team.id`         |
| `conference` | String    | The conference the team belongs to (e.g., 'East', 'West').        | Main field. API: `team.conference` |
| `division`   | String    | The division the team belongs to (e.g., 'Southeast', 'Pacific').  | Main field. API: `team.division`   |
| `city`       | String    | The city where the team is based (e.g., 'Atlanta', 'Boston').     | Main field. API: `team.city`       |
| `name`       | String    | The team's name (e.g., 'Hawks', 'Celtics').                       | Main field. API: `team.name`       |
| `full_name`  | String    | The full name of the team (e.g., 'Atlanta Hawks').                | Main field. API: `team.full_name`  |
| `abbreviation`| String   | The common abbreviation for the team (e.g., 'ATL', 'BOS').        | Main field. API: `team.abbreviation`|



### Note: Data was extracted using API hosted at BallDontLie.io, GOAT subscription plan required to connect.