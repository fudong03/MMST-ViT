import os
import json
import pandas as pd


def get_counties():
    file_path = "./../input/county_info.csv"
    df = pd.read_csv(file_path)

    # convert state name to its abbreviation
    state_abbreviation = get_state_abbreviation()
    df['State'] = df['State'].map(state_abbreviation)

    df['FIPS'] = df['FIPS Code'].astype(str)
    counties = df.to_json(orient='records')

    return json.loads(counties)


def get_state_abbreviation():
    state_abbreviations = {
        'ALABAMA': 'AL',
        'ALASKA': 'AK',
        'ARIZONA': 'AZ',
        'ARKANSAS': 'AR',
        'CALIFORNIA': 'CA',
        'COLORADO': 'CO',
        'CONNECTICUT': 'CT',
        'DELAWARE': 'DE',
        'FLORIDA': 'FL',
        'GEORGIA': 'GA',
        'HAWAII': 'HI',
        'IDAHO': 'ID',
        'ILLINOIS': 'IL',
        'INDIANA': 'IN',
        'IOWA': 'IA',
        'KANSAS': 'KS',
        'KENTUCKY': 'KY',
        'LOUISIANA': 'LA',
        'MAINE': 'ME',
        'MARYLAND': 'MD',
        'MASSACHUSETTS': 'MA',
        'MICHIGAN': 'MI',
        'MINNESOTA': 'MN',
        'MISSISSIPPI': 'MS',
        'MISSOURI': 'MO',
        'MONTANA': 'MT',
        'NEBRASKA': 'NE',
        'NEVADA': 'NV',
        'NEW HAMPSHIRE': 'NH',
        'NEW JERSEY': 'NJ',
        'NEW MEXICO': 'NM',
        'NEW YORK': 'NY',
        'NORTH CAROLINA': 'NC',
        'NORTH DAKOTA': 'ND',
        'OHIO': 'OH',
        'OKLAHOMA': 'OK',
        'OREGON': 'OR',
        'PENNSYLVANIA': 'PA',
        'RHODE ISLAND': 'RI',
        'SOUTH CAROLINA': 'SC',
        'SOUTH DAKOTA': 'SD',
        'TENNESSEE': 'TN',
        'TEXAS': 'TX',
        'UTAH': 'UT',
        'VERMONT': 'VT',
        'VIRGINIA': 'VA',
        'WASHINGTON': 'WA',
        'WEST VIRGINIA': 'WV',
        'WISCONSIN': 'WI',
        'WYOMING': 'WY'
    }

    return state_abbreviations


def build_one_county(year, county_info):
    fips, county, state = county_info["FIPS"], county_info["County"], county_info["State"]

    obj = {
        "FIPS": fips,
        "year": int(year),
        "county": county,
        "state": state,
        "county_ansi": fips[2:],
        "state_ansi": fips[:2],
        "data": {
            "HRRR": {
                "short_term": build_short_term(fips, state, year),
                "long_term": build_long_term(fips, state, year, offset=5)
            },
            "USDA": "USDA/data/Soybeans/{}/USDA_Soybeans_County_{}.csv".format(year, year),
            "sentinel": build_sentinel(fips, state, year, seasons=["Summer", "Fall"]),
        },
    }

    return obj


def build_short_term(fips, state, year, start_month=4, end_month=9):
    short_term = []
    for month in range(start_month, end_month + 1):
        file_name = "HRRR/data/{}/{}/HRRR_{}_{}_{}-{}.csv".format(year, state, fips[:2], state, year,
                                                                  '{:02d}'.format(month))
        short_term.append(file_name)
    return short_term


def build_long_term(fips, state, cur_year, offset=5):
    long_term = []
    for past_year in range(cur_year - offset, cur_year):
        past_year_data = []
        for month in range(1, 13):
            file_path = "HRRR/data/{}/{}/HRRR_{}_{}_{}-{}.csv".format(past_year, state, fips[:2], state, past_year,
                                                                      '{:02d}'.format(month)),
            past_year_data.append(file_path)
        long_term.append(past_year_data)
    return long_term


def build_sentinel(fips, state, cur_year, seasons=["Summer", "Fall"]):
    season_dic = {
        "Spring": ["01-01", "03-31"],
        "Summer": ["04-01", "06-30"],
        "Fall": ["07-01", "09-30"],
        "Winter": ["10-01", "12-31"],
    }

    sentinel = []
    for season in seasons:
        # e.g., "04-01"
        start_day, end_day = season_dic[season][0], season_dic[season][1]
        # e.g., "2021-04-01"
        start_date = str(cur_year) + "-" + start_day
        end_date = str(cur_year) + "-" + end_day

        file_path = "Sentinel/data/AG/{}/{}/Agriculture_{}_{}_{}_{}.h5" \
                        .format(cur_year, state, fips[:2], state, start_date, end_date)
        sentinel.append(file_path)

    return sentinel


def build_train():
    root = "./../data"
    file_path = "soybean_train.json"
    path = os.path.join(root, file_path)

    years = [2021]
    counties = get_counties()

    data = []
    for year in years:
        for county_info in counties:
            obj = build_one_county(year, county_info)
            data.append(obj)

    with open(path, "x") as write_file:
        # write the data to the file in JSON format
        json.dump(data, write_file)


def build_val():
    root = "./../data"
    file_path = "soybean_val.json"
    path = os.path.join(root, file_path)

    years = [2022]
    counties = get_counties()

    data = []
    for year in years:
        for county_info in counties:
            obj = build_one_county(year, county_info)
            data.append(obj)

    with open(path, "x") as write_file:
        # write the data to the file in JSON format
        json.dump(data, write_file)


if __name__ == '__main__':
    build_train()
    build_val()
