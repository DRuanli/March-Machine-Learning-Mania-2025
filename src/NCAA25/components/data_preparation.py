import glob
import numpy as np
import pandas as pd
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from src.NCAA25 import logger
from src.NCAA25.entity import DataPreparationConfig
from src.NCAA25.utils.common import save_dataframe, save_model


class DataPreparation:
    def __init__(self, config: DataPreparationConfig):
        self.config = config
        self.data = {}
        self.teams = None
        self.seeds = None
        self.games = None
        self.features = None
        self.col = None  # Feature columns

        # Preprocessing objects
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load all CSV files from the data directory
        """
        try:
            # Check directory structure
            logger.info(f"Checking directory structure at {self.config.data_path}")
            for root, dirs, files in os.walk(self.config.data_path):
                logger.info(f"Found directory: {root}")
                logger.info(f"Contains subdirectories: {dirs}")
                logger.info(f"Contains files: {len(files)}")
                if files:
                    logger.info(f"Sample files: {files[:5]}")

            # Update path pattern to search recursively
            files = glob.glob(os.path.join(self.config.data_path, "**/*.csv"), recursive=True)

            if not files:
                logger.error(f"No CSV files found in {self.config.data_path} or subdirectories")
                raise FileNotFoundError(f"No CSV files found in {self.config.data_path}")

            self.data = {
                os.path.basename(p).split('.')[0]: pd.read_csv(p, encoding='latin-1')
                for p in files
            }

            logger.info(f"Loaded {len(self.data)} CSV files: {list(self.data.keys())}")
            return self.data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise e

    def process_teams(self):
        """
        Process teams and team spellings data
        """
        try:
            # Process teams
            teams = pd.concat([self.data.get('MTeams', pd.DataFrame()),
                               self.data.get('WTeams', pd.DataFrame())])

            # Process team spellings
            teams_spelling = pd.concat([self.data.get('MTeamSpellings', pd.DataFrame()),
                                        self.data.get('WTeamSpellings', pd.DataFrame())])

            if not teams_spelling.empty:
                teams_spelling = teams_spelling.groupby(by='TeamID', as_index=False)['TeamNameSpelling'].count()
                teams_spelling.columns = ['TeamID', 'TeamNameCount']
                self.teams = pd.merge(teams, teams_spelling, how='left', on=['TeamID'])
            else:
                self.teams = teams

            logger.info(f"Processed teams data: {len(self.teams)} teams")

            # Save processed teams data
            save_dataframe(self.teams, os.path.join(self.config.output_dir, "teams_processed.csv"))

            return self.teams
        except Exception as e:
            logger.error(f"Error processing teams: {e}")
            raise e

    def process_tournament_seeds(self):
        """
        Process tournament seeds data
        """
        try:
            # Load seeds data
            seeds_df = pd.concat([self.data.get('MNCAATourneySeeds', pd.DataFrame()),
                                  self.data.get('WNCAATourneySeeds', pd.DataFrame())])

            # Convert seeds dataframe to a dictionary
            self.seeds = {
                '_'.join(map(str, [int(k1), k2])): int(v[1:3])
                for k1, v, k2 in seeds_df[['Season', 'Seed', 'TeamID']].values
            }

            logger.info(f"Processed tournament seeds: {len(self.seeds)} seed entries")

            # Save as a dataframe for later use
            seeds_processed = pd.DataFrame([
                {'key': k, 'seed': v} for k, v in self.seeds.items()
            ])
            save_dataframe(seeds_processed, os.path.join(self.config.output_dir, "seeds_processed.csv"))

            return self.seeds
        except Exception as e:
            logger.error(f"Error processing tournament seeds: {e}")
            raise e

    def process_game_results(self):
        """
        Process regular season and tournament game results
        """
        try:
            # Concatenate game results data
            season_dresults = pd.concat([self.data.get('MRegularSeasonDetailedResults', pd.DataFrame()),
                                         self.data.get('WRegularSeasonDetailedResults', pd.DataFrame())])
            tourney_dresults = pd.concat([self.data.get('MNCAATourneyDetailedResults', pd.DataFrame()),
                                          self.data.get('WNCAATourneyDetailedResults', pd.DataFrame())])

            # Mark results as season or tournament
            season_dresults['ST'] = 'S'
            tourney_dresults['ST'] = 'T'

            # Combine all data
            self.games = pd.concat((season_dresults, tourney_dresults), axis=0, ignore_index=True)
            self.games.reset_index(drop=True, inplace=True)

            # Process WLoc as categorical
            self.games['WLoc'] = self.games['WLoc'].map({'A': 1, 'H': 2, 'N': 3})

            # Create game identifiers and team-related features
            self.games['ID'] = self.games.apply(
                lambda r: '_'.join(map(str, [r['Season']] + sorted([r['WTeamID'], r['LTeamID']]))), axis=1
            )
            self.games['IDTeams'] = self.games.apply(
                lambda r: '_'.join(map(str, sorted([r['WTeamID'], r['LTeamID']]))), axis=1
            )
            self.games['Team1'] = self.games.apply(
                lambda r: sorted([r['WTeamID'], r['LTeamID']])[0], axis=1
            )
            self.games['Team2'] = self.games.apply(
                lambda r: sorted([r['WTeamID'], r['LTeamID']])[1], axis=1
            )
            self.games['IDTeam1'] = self.games.apply(
                lambda r: '_'.join(map(str, [r['Season'], r['Team1']])), axis=1
            )
            self.games['IDTeam2'] = self.games.apply(
                lambda r: '_'.join(map(str, [r['Season'], r['Team2']])), axis=1
            )

            # Add seed-related features
            self.games['Team1Seed'] = self.games['IDTeam1'].map(self.seeds).fillna(0)
            self.games['Team2Seed'] = self.games['IDTeam2'].map(self.seeds).fillna(0)

            # Add additional features
            self.games['ScoreDiff'] = self.games['WScore'] - self.games['LScore']
            self.games['Pred'] = self.games.apply(
                lambda r: 1.0 if sorted([r['WTeamID'], r['LTeamID']])[0] == r['WTeamID'] else 0.0, axis=1
            )
            self.games['ScoreDiffNorm'] = self.games.apply(
                lambda r: r['ScoreDiff'] * -1 if r['Pred'] == 0.0 else r['ScoreDiff'], axis=1
            )
            self.games['SeedDiff'] = self.games['Team1Seed'] - self.games['Team2Seed']

            # Fill missing values
            self.games = self.games.fillna(-1)

            logger.info(f"Processed game results: {len(self.games)} games")

            # Save processed game data
            save_dataframe(self.games, os.path.join(self.config.output_dir, "games_processed.csv"))

            return self.games
        except Exception as e:
            logger.error(f"Error processing game results: {e}")
            raise e

    def aggregate_statistics(self):
        """
        Aggregate game statistics by team pairing
        """
        try:
            # Define statistics columns to aggregate
            c_score_col = [
                'NumOT', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst',
                'WTO', 'WStl', 'WBlk', 'WPF', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA',
                'LOR', 'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF'
            ]

            # Define aggregation functions
            c_score_agg = ['sum', 'mean', 'median', 'max', 'min', 'std']

            # Aggregate statistics by team pairing
            gb = self.games.groupby(by=['IDTeams']).agg({k: c_score_agg for k in c_score_col}).reset_index()
            gb.columns = [''.join(c) + '_c_score' for c in gb.columns]

            logger.info(f"Aggregated statistics for {len(gb)} team pairings")

            # Save aggregated statistics
            save_dataframe(gb, os.path.join(self.config.output_dir, "aggregated_stats.csv"))

            # Filter to tournament games for training
            tourney_games = self.games[self.games['ST'] == 'T']

            # Merge aggregated stats into games
            tourney_games = pd.merge(
                tourney_games,
                gb,
                how='left',
                left_on='IDTeams',
                right_on='IDTeams_c_score'
            )

            logger.info(f"Prepared {len(tourney_games)} tournament games with statistics")

            # Define feature columns
            exclude_cols = [
                               'ID', 'DayNum', 'ST', 'Team1', 'Team2', 'IDTeams', 'IDTeam1', 'IDTeam2',
                               'WTeamID', 'WScore', 'LTeamID', 'LScore', 'NumOT', 'Pred', 'ScoreDiff',
                               'ScoreDiffNorm', 'WLoc'
                           ] + c_score_col

            self.col = [c for c in tourney_games.columns if c not in exclude_cols]

            # Save feature columns list
            pd.DataFrame({'feature': self.col}).to_csv(
                os.path.join(self.config.output_dir, "feature_columns.csv"),
                index=False
            )

            # Save tournament games with features
            save_dataframe(tourney_games, os.path.join(self.config.output_dir, "tourney_games_features.csv"))

            return tourney_games, self.col
        except Exception as e:
            logger.error(f"Error aggregating statistics: {e}")
            raise e

    def fit_preprocessors(self, data):
        """
        Fit the imputer and scaler on the data

        Args:
            data (pd.DataFrame): DataFrame with features

        Returns:
            tuple: (imputer, scaler)
        """
        try:
            X = data[self.col].fillna(-1)

            # Fit the imputer and scaler
            self.imputer.fit(X)
            X_imputed = self.imputer.transform(X)
            self.scaler.fit(X_imputed)

            # Save the preprocessors
            save_model(self.imputer, os.path.join(self.config.output_dir, "imputer.joblib"))
            save_model(self.scaler, os.path.join(self.config.output_dir, "scaler.joblib"))

            logger.info("Fit and saved preprocessors")

            return self.imputer, self.scaler
        except Exception as e:
            logger.error(f"Error fitting preprocessors: {e}")
            raise e

    def run(self):
        """
        Run the complete data preparation pipeline
        """
        try:
            logger.info("Starting data preparation process")

            # Load data
            self.load_data()

            # Process teams and seeds
            self.process_teams()
            self.process_tournament_seeds()

            # Process game results
            self.process_game_results()

            # Aggregate statistics and prepare features
            tourney_games, _ = self.aggregate_statistics()

            # Fit preprocessors
            self.fit_preprocessors(tourney_games)

            logger.info("Data preparation completed successfully")

            return True
        except Exception as e:
            logger.error(f"Error in data preparation: {e}")
            raise e