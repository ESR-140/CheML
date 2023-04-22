import pandas as pd

def get_features(target, all_features='No'):

    if all_features == 'Yes':
        feature = ['Eg', 'Eg(HSE)','DOS', 'Et', 'EFermi', 'SosX', 'SosY', 'Ew', 'SePx', 'SePy', 'Dir Eg', 'St11com', '2DpfX','2DpfY', 'AuCell']
        feature.remove(target)
    else:
        data = {"Target": ['AuCell', 'Eg', 'Eg(HSE)', 'Et', 'Ew', 'EFermi'],
                "Features": [['Et', 'St11com', 'SosY', 'SosX', 'Eg', 'SePx', 'HoF', 'Eg(HSE)', 'EFermi', 'Dir Eg', 'SePy'],
                            ['SePx', 'HoF', 'Et', 'SePy', '2DpfY', 'SosY', 'Ew', 'St11com', 'EFermi'],
                            ['SePx', 'SePy', 'HoF', 'SosY', '2DpfX', '2DpfY', 'Ew', 'EFermi', 'St11com', 'Et', 'AuCell'],
                            ['AuCell', 'St11com', 'SosY', 'HoF', 'Eg', 'EFermi', 'SosX', 'SePy', 'Eg(HSE)'],
                            ['EFermi', 'St11com', 'SePy', 'Et', 'HoF', 'SosY', 'SePx', 'AuCell', 'SosX'],
                            ['Ew', 'St11com', 'SePy', 'SePx', 'Et', 'AuCell', 'Eg(HSE)', 'SosX', 'SosY'],
                            ]}

        df = pd.DataFrame(data)
        row = df.loc[df["Target"] == target]
        feature = row["Features"].values[0]

    return feature
