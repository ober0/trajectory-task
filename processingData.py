from sqlalchemy import create_engine
import pandas as pd

class ProcessingData:
    def __init__(self):
        self.engine = create_engine('sqlite:///trajectory.db')

    def parseData(self, path: str):
        df = pd.read_excel(f'files/input/{path}')
        df = df.iloc[:, [0, 1]]
        df.columns.values[0] = 'date'
        df.columns.values[1] = 'temp'
        df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y %H:%M')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['temp'] = df.groupby('month')['temp'].transform(lambda x: x.fillna(x.mean()))
        df['year'] = df['year'].fillna(df['year'].mode()[0]).astype(int)
        df['month'] = df['month'].fillna(df['month'].mode()[0]).astype(int)
        monthly_avg_temp = df.groupby(['year', 'month'])['temp'].mean().reset_index()
        monthly_avg_temp.to_sql('monthly_weather_data', con=self.engine, if_exists='replace', index=False)

    def get_weather_data(self) -> pd.DataFrame:
        query = "SELECT year, month, temp FROM monthly_weather_data"
        df = pd.read_sql(query, con=self.engine)
        return df

    def get_latest_date(self) -> str:
        query = "SELECT MAX(year) AS max_year FROM monthly_weather_data"
        result = pd.read_sql(query, con=self.engine)
        latest_year = result['max_year'].iloc[0]

        query = f"SELECT MAX(month) AS max_month FROM monthly_weather_data WHERE year = {latest_year}"
        result = pd.read_sql(query, con=self.engine)
        latest_month = result['max_month'].iloc[0]

        return f"{latest_year}-{latest_month:02d}-01"

