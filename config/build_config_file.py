import os
import json


def get_counties():
    counties = [
        {"FIPS": "01003",
         "county": "BALDWIN",
         "state": "AL",
         },
        # {"FIPS": "22007",
        #  "county": "ASSUMPTION",
        #  "state": "LA",
        #  },
        # {"FIPS": "22121",
        #  "county": "WEST BATON ROUGE",
        #  "state": "LA",
        #  },
        # {"FIPS": "28123",
        #  "county": "SCOTT",
        #  "state": "MS",
        #  },
        # {"FIPS": "28089",
        #  "county": "WMADISON",
        #  "state": "MS",
        #  },
        # {"FIPS": "17091",
        #  "county": "LAKE",
        #  "state": "IL",
        #  },
        # {"FIPS": "17155",
        #  "county": "PUTNAM",
        #  "state": "IL",
        #  },
        # {"FIPS": "19117",
        #  "county": "LUCAS",
        #  "state": "IA",
        #  },
        # {"FIPS": "19135",
        #  "county": "MONROE",
        #  "state": "IA",
        #  },
    ]

    return counties


def get_prev_year(year, offset):
    return str(int(year) - offset)


def get_json_obj(year, county_info):
    fips, county, state = county_info["FIPS"], county_info["county"], county_info["state"]

    obj = {
        "FIPS": fips,
        "year": int(year),
        "county": county,
        "state": state,
        "county_ansi": fips[2:],
        "state_ansi": fips[:2],
        "data": {
            "HRRR": {
                # "short_term": [
                #     "HRRR/data/{}/{}/HRRR_{}_{}_{}-04.csv".format(year, state, fips[:2], state, year),
                #     "HRRR/data/{}/{}/HRRR_{}_{}_{}-05.csv".format(year, state, fips[:2], state, year),
                #     "HRRR/data/{}/{}/HRRR_{}_{}_{}-06.csv".format(year, state, fips[:2], state, year),
                #     "HRRR/data/{}/{}/HRRR_{}_{}_{}-07.csv".format(year, state, fips[:2], state, year),
                #     "HRRR/data/{}/{}/HRRR_{}_{}_{}-08.csv".format(year, state, fips[:2], state, year),
                #     "HRRR/data/{}/{}/HRRR_{}_{}_{}-09.csv".format(year, state, fips[:2], state, year),
                # ],
                "short_term": [
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-01.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-02.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-03.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-04.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-05.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-06.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-07.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-08.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-09.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-10.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-11.csv".format(year, state, fips[:2], state, year),
                    "HRRR/data/{}/{}/HRRR_{}_{}_{}-12.csv".format(year, state, fips[:2], state, year),
                ],
                "long_term": [
                    [
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-01.csv".format(str(int(year) - 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-02.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-03.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-04.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-05.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-06.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-07.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-08.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-09.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-10.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-11.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-12.csv".format(get_prev_year(year, 1), state, fips[:2],
                                                                      state, get_prev_year(year, 1)),
                    ],
                    [
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-01.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-02.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-03.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-04.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-05.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-06.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-07.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-08.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-09.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-10.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-11.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-12.csv".format(get_prev_year(year, 2), state, fips[:2],
                                                                      state, get_prev_year(year, 2)),
                    ],
                    [
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-01.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-02.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-03.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-04.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-05.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-06.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-07.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-08.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-09.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-10.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-11.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-12.csv".format(get_prev_year(year, 3), state, fips[:2],
                                                                      state, get_prev_year(year, 3)),
                    ],
                    [
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-01.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-02.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-03.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-04.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-05.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-06.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-07.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-08.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-09.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-10.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-11.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-12.csv".format(get_prev_year(year, 4), state, fips[:2],
                                                                      state, get_prev_year(year, 4)),
                    ],
                    [
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-01.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-02.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-03.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-04.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-05.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-06.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-07.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-08.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-09.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-10.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-11.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                        "HRRR/data/{}/{}/HRRR_{}_{}_{}-12.csv".format(get_prev_year(year, 5), state, fips[:2],
                                                                      state, get_prev_year(year, 5)),
                    ],

                ]
            },
            "USDA": "USDA/data/Soybeans/{}/USDA_Soybean_County_{}.csv".format(year, year),
            "sentinel": [
                "Sentinel/data/AG/{}/{}/Agriculture_{}_{}_{}-04-01_{}-06-30.h5".format(year, state, fips[:2], state, year, year),
                "Sentinel/data/AG/{}/{}/Agriculture_{}_{}_{}-07-01_{}-09-30.h5".format(year, state, fips[:2], state,
                                                                                       year, year),
            ]
        }

    }
    return obj


def build_train():
    root = "/mnt/share/github/crop_prediction/data"
    file_path = "soybean_train.json"
    path = os.path.join(root, file_path)

    years = ["2020"]
    counties = get_counties()

    data = []
    for year in years:
        for county_info in counties:
            obj = get_json_obj(year, county_info)
            data.append(obj)

    with open(path, "x") as write_file:
        # write the data to the file in JSON format
        json.dump(data, write_file)


def build_val():
    root = "/mnt/share/github/crop_prediction/data"
    file_path = "soybean_val.json"
    path = os.path.join(root, file_path)

    years = ["2022"]
    counties = get_counties()

    data = []
    for year in years:
        for county_info in counties:
            obj = get_json_obj(year, county_info)
            data.append(obj)

    with open(path, "x") as write_file:
        # write the data to the file in JSON format
        json.dump(data, write_file)


if __name__ == '__main__':
    # build_train()
    build_val()
