
DHS_COUNTRIES = [
    'angola', 'benin', 'burkina_faso', 'cote_d_ivoire',
    'democratic_republic_of_congo', 'ethiopia', 'ghana', 'guinea', 'kenya',
    'lesotho', 'malawi', 'mozambique', 'nigeria', 'rwanda', 'senegal',
    'sierra_leone', 'tanzania', 'togo', 'uganda', 'zambia', 'zimbabwe']

LSMS_COUNTRIES = ['ethiopia', 'malawi', 'nigeria', 'tanzania', 'uganda']

_SURVEY_NAMES_5country = {
    'train': ['uganda_2011', 'tanzania_2010', 'rwanda_2010', 'nigeria_2013'],
    'val': ['malawi_2010']
}

_SURVEY_NAMES_DHS_OOC_A = {
    'train': [ 'democratic_republic_of_congo', 'ghana', 'kenya',
              'lesotho', 'malawi', 'mozambique', 'nigeria', 'senegal',
              'togo', 'uganda', 'zambia', 'zimbabwe'],
    'val': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
    'test': ['angola', 'cote_d_ivoire', 'ethiopia', 'rwanda'],
}
_SURVEY_NAMES_DHS_OOC_B = {
    'train': ['angola', 'cote_d_ivoire', 'democratic_republic_of_congo',
              'ethiopia', 'kenya', 'lesotho', 'mozambique',
              'nigeria', 'rwanda', 'senegal', 'togo', 'uganda', 'zambia'],
    'val': [ 'ghana', 'malawi', 'zimbabwe'],
    'test': ['benin', 'burkina_faso', 'guinea', 'sierra_leone', 'tanzania'],
}
_SURVEY_NAMES_DHS_OOC_C = {
    'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire', 'ethiopia',
              'guinea', 'kenya', 'lesotho', 'rwanda', 'senegal',
              'sierra_leone', 'tanzania', 'zambia'],
    'val': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
    'test': [ 'ghana', 'malawi', 'zimbabwe'],
}
_SURVEY_NAMES_DHS_OOC_D = {
    'train': ['angola', 'benin', 'burkina_faso', 'cote_d_ivoire',
              'ethiopia', 'ghana', 'guinea', 'malawi', 'rwanda',
              'sierra_leone', 'tanzania', 'zimbabwe'],
    'val': ['kenya', 'lesotho', 'senegal', 'zambia'],
    'test': ['democratic_republic_of_congo', 'mozambique', 'nigeria', 'togo', 'uganda'],
}
_SURVEY_NAMES_DHS_OOC_E = {
    'train': ['benin', 'burkina_faso', 'democratic_republic_of_congo',
              'ghana', 'guinea', 'malawi', 'mozambique', 'nigeria', 'sierra_leone',
              'tanzania', 'togo', 'uganda', 'zimbabwe'],
    'val': ['angola', 'cote_d_ivoire', 'ethiopia', 'rwanda'],
    'test': ['kenya', 'lesotho', 'senegal', 'zambia'],
}

SURVEY_NAMES = {  # TODO: rename to SURVEY_NAMES_DHS?
    '5country': _SURVEY_NAMES_5country,  # TODO: is this needed
    'DHS_OOC_A': _SURVEY_NAMES_DHS_OOC_A,
    'DHS_OOC_B': _SURVEY_NAMES_DHS_OOC_B,
    'DHS_OOC_C': _SURVEY_NAMES_DHS_OOC_C,
    'DHS_OOC_D': _SURVEY_NAMES_DHS_OOC_D,
    'DHS_OOC_E': _SURVEY_NAMES_DHS_OOC_E
}

SURVEY_NAMES_LSMS = ['ethiopia_2011', 'ethiopia_2015', 'malawi_2010', 'malawi_2016',
                      'nigeria_2010', 'nigeria_2015', 'tanzania_2008', 'tanzania_2012',
                      'uganda_2005', 'uganda_2009', 'uganda_2013']

SIZES = {
    'DHS': {'train': 12319, 'val': 3257, 'test': 4093, 'all': 19669},  # TODO: is this needed? original is + 192 for train and all
    'DHSNL': {'all': 260415},
    'DHS_OOC_A': {'train': 11797, 'val': 3909, 'test': 3963, 'all': 19669},
    'DHS_OOC_B': {'train': 11820, 'val': 3940, 'test': 3909, 'all': 19669},
    'DHS_OOC_C': {'train': 11800, 'val': 3929, 'test': 3940, 'all': 19669},
    'DHS_OOC_D': {'train': 11812, 'val': 3928, 'test': 3929, 'all': 19669},
    'DHS_OOC_E': {'train': 11778, 'val': 3963, 'test': 3928, 'all': 19669},
    'DHS_incountry_A': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_B': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_C': {'train': 11801, 'val': 3934, 'test': 3934, 'all': 19669},
    'DHS_incountry_D': {'train': 11802, 'val': 3933, 'test': 3934, 'all': 19669},
    'DHS_incountry_E': {'train': 11802, 'val': 3934, 'test': 3933, 'all': 19669},
    'LSMSincountry': {'train': 1812, 'val': 604, 'test': 604, 'all': 3020},  # TODO: is this needed?
    'LSMS': {'ethiopia_2011': 327, 'ethiopia_2015': 327, 'malawi_2010': 102,
             'malawi_2016': 102, 'nigeria_2010': 480, 'nigeria_2015': 480,
             'tanzania_2008': 300, 'tanzania_2012': 300, 'uganda_2005': 165,
             'uganda_2009': 165, 'uganda_2013': 165},
}

URBAN_SIZES = {
    'DHS': {'train': 3954, 'val': 1212, 'test': 1635, 'all': 6801},
    'DHS_OOC_A': {'train': 4264, 'val': 1221, 'test': 1316, 'all': 6801},
    'DHS_OOC_B': {'train': 4225, 'val': 1355, 'test': 1221, 'all': 6801},
    'DHS_OOC_C': {'train': 4010, 'val': 1436, 'test': 1355, 'all': 6801},
    'DHS_OOC_D': {'train': 3892, 'val': 1473, 'test': 1436, 'all': 6801},
    'DHS_OOC_E': {'train': 4012, 'val': 1316, 'test': 1473, 'all': 6801},
}

RURAL_SIZES = {
    'DHS': {'train': 8365, 'val': 2045, 'test': 2458, 'all': 12868},
    'DHS_OOC_A': {'train': 7533, 'val': 2688, 'test': 2647, 'all': 12868},
    'DHS_OOC_B': {'train': 7595, 'val': 2585, 'test': 2688, 'all': 12868},
    'DHS_OOC_C': {'train': 7790, 'val': 2493, 'test': 2585, 'all': 12868},
    'DHS_OOC_D': {'train': 7920, 'val': 2455, 'test': 2493, 'all': 12868},
    'DHS_OOC_E': {'train': 7766, 'val': 2647, 'test': 2455, 'all': 12868},
}
_MAX_DHS={
    'BLUE':2.0,
    'GREEN': 2.0,
    'RED': 2.0,
    'SWIR1': 2.0,
    'SWIR2':316.29,
    'TEMP1':2.0

}
# means and standard deviations calculated over the entire dataset (train + val + test),
# with negative values set to 0, and ignoring any pixel that is 0 across all bands
#'maxs': array([2.00000000e+00, 2.00000000e+00, 2.00000000e+00, 2.00000000e+00,   2.00000000e+00, 3.16299988e+02, 2.00000000e+00, 1.17869397e+03,   2.52104688e+03])
#min :'mins_nz': array([9.99999975e-05, 4.10000002e-03, 9.99999975e-05, 6.50000002e-04,  7.50000007e-04, 2.73450012e+02, 6.25000009e-03, 2.93550444e+00,    8.53324309e-08]), 'mins_goodpx': array([-1.12599999e-01, -9.99999975e-05, -2.43999995e-02, -2.00000009e-03,  -3.10000009e-03,  0.00000000e+00, -6.30000001e-03,  0.00000000e+00,     -9.60280448e-02]),
#'mins': array([-1.12599999e-01,-9.99999975e-05, -2.43999995e-02, -2.00000009e-03,    -3.10000009e-03,  0.00000000e+00, -6.30000001e-03,  0.00000000e+00,   -9.60280448e-02])
_MEANS_DHS = {
    'BLUE':  0.059183,
    'GREEN': 0.088619,
    'RED':   0.104145,
    'SWIR1': 0.246874,
    'SWIR2': 0.168728,
    'TEMP1': 299.078023,
    'NIR':   0.253074,
    'DMSP':  4.005496,
    'VIIRS': 1.096089,
    # 'NIGHTLIGHTS': 5.101585, # nightlights overall
}

#{'BLUE': 0.05822182047376457, 'GREEN': 0.0875679257877037, 'RED': 0.10238159792927382, 'SWIR1': 0.2447995798290326, 'SWIR2': 0.1667005879554217, 'TEMP1': 298.9744695004072, 'NIR': 0.2520130062401763, 'DMSP': 3.8304217292437572, 'VIIRS': 1.1020250701140772}
#{'BLUE': 0.022382626993272824, 'GREEN': 0.031043231871274456, 'RED': 0.05007785714003174, 'SWIR1': 0.08682955619110806, 'SWIR2': 0.0812648754057452, 'TEMP1': 4.077351547622513, 'NIR': 0.05879342933901235, 'DMSP': 23.06379172055441, 'VIIRS': 4.850151399753214}

_MEANS_DHSNL = {
    'BLUE':  0.063927,
    'GREEN': 0.091981,
    'RED':   0.105234,
    'SWIR1': 0.235316,
    'SWIR2': 0.162268,
    'TEMP1': 298.736746,
    'NIR':   0.245430,
    'DMSP':  7.152961,
    'VIIRS': 2.322687,
}
#{'BLUE': 0.057537556172876786, 'GREEN': 0.08651898793547044, 'RED': 0.1007862721524, 'SWIR1': 0.242318225317176, 'SWIR2': 0.16401278741135097, 'TEMP1': 298.8947231903005, 'NIR': 0.2505241268481043, 'DMSP': 3.400936207242984, 'VIIRS': 0.9495341304385304}
#{'BLUE': 0.0221962404313549, 'GREEN': 0.03102870864122559, 'RED': 0.05024759088452628, 'SWIR1': 0.0890930364869842, 'SWIR2': 0.08224543155125386, 'TEMP1': 4.08558551573984, 'NIR': 0.061624549064828976, 'DMSP': 21.37118096584169, 'VIIRS': 4.793903106829549}
#reshaping
_MEANS_LSMS = {
    'BLUE':  0.062551,
    'GREEN': 0.090696,
    'RED':   0.105640,
    'SWIR1': 0.242577,
    'SWIR2': 0.165792,
    'TEMP1': 299.495280,
    'NIR':   0.256701,
    'DMSP':  5.105815,
    'VIIRS': 0.557793,
}

_STD_DEVS_DHS = {
    'BLUE':  0.022926,
    'GREEN': 0.031880,
    'RED':   0.051458,
    'SWIR1': 0.088857,
    'SWIR2': 0.083240,
    'TEMP1': 4.300303,
    'NIR':   0.058973,
    'DMSP':  23.038301,
    'VIIRS': 4.786354,
    # 'NIGHTLIGHTS': 23.342916, # nightlights overall
}
_STD_DEVS_DHSNL = {
    'BLUE':  0.023697,
    'GREEN': 0.032474,
    'RED':   0.051421,
    'SWIR1': 0.095830,
    'SWIR2': 0.087522,
    'TEMP1': 6.208949,
    'NIR':   0.071084,
    'DMSP':  29.749457,
    'VIIRS': 14.611589,
}
_STD_DEVS_LSMS = {
    'BLUE':  0.023979,
    'GREEN': 0.032121,
    'RED':   0.051943,
    'SWIR1': 0.088163,
    'SWIR2': 0.083826,
    'TEMP1': 4.678959,
    'NIR':   0.059025,
    'DMSP':  31.688320,
    'VIIRS': 6.421816,
}

MEANS_DICT = {
    'DHS': _MEANS_DHS,
    'DHSNL': _MEANS_DHSNL,
    'LSMS': _MEANS_LSMS,
}

STD_DEVS_DICT = {
    'DHS': _STD_DEVS_DHS,
    'DHSNL': _STD_DEVS_DHSNL,
    'LSMS': _STD_DEVS_LSMS,
}
