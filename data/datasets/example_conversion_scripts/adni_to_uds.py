import pandas as pd
import numpy as np
import re

def moca(df):
    """
    computes domain scores for moca.
    run this before applying uds conversion.
    """
    # naming
    df['MOCANAMI'] =  df[['LION',  'RHINO', 'CAMEL']].sum(axis = 1)
    # memory registration
    df['MOCAREGI'] = df[['IMMT1W1', 'IMMT1W2',
       'IMMT1W3', 'IMMT1W4', 'IMMT1W5', 'IMMT2W1', 'IMMT2W2', 'IMMT2W3',
       'IMMT2W4', 'IMMT2W5']].sum(axis=1)
    df['MOCADIGI'] = df[['DIGFOR', 'DIGBACK']].sum(axis=1)
    # letters
    df['MOCALETT'] = 0  # Default to 0
    df.loc[(moca['LETTERS'] == 0) | (moca['LETTERS'] == 1), 'MOCALETT'] = 1
    # serial 7
    moca_score = df[['SERIAL1', 'SERIAL2', 'SERIAL3', 'SERIAL4', 'SERIAL5']].sum(axis=1)
    df['MOCASER7'] = 0  # Default to 0
    df.loc[moca_score == 1, 'MOCASER7'] = 1
    df.loc[(moca_score == 2) | (moca_score == 3), 'MOCASER7'] = 2
    df.loc[(moca_score == 4) | (moca_score == 5), 'MOCASER7'] = 3
    # language
    df['MOCAREPE'] = df[['REPEAT1', 'REPEAT2']].sum(axis=1)
    # fluency
    df['MOCAFLUE'] = 0  # Default to 0
    df.loc[moca['FFLUENCY'] >= 11, 'MOCAFLUE'] = 1    
    # abstraction
    df['MOCAABST'] = df[['ABSTRAN', 'ABSMEAS']].sum(axis=1)
    # delayed recall
    #MoCA: Delayed recall — No cue
    df['MOCARECN'] =  df[['DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5']].eq(1).sum(axis=1)
    #MoCA: Delayed recall — Category cue
    df['MOCARECC'] = df[['DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5']].eq(2).sum(axis=1)
    #MoCA: Delayed recall — Recognition
    df['MOCARECR'] = df[['DELW1', 'DELW2', 'DELW3', 'DELW4', 'DELW5']].eq(3).sum(axis=1)
    return df


def npiq(df):
    # Recoding NPIQ to match ADRD
    conditions = {
        'DEL': 'DELSEV',
        'HALL': 'HALLSEV',
        'AGIT': 'AGITSEV',
        'DEPD': 'DEPDSEV',
        'ANX' : 'ANXSEV',
        'ELAT': 'ELATSEV',
        'APA': 'APASEV',
        'DISN': 'DISNSEV',
        'IRR': 'IRRSEV', 
        'MOT': 'MOTSEV',
        'NITE': 'NITESEV',
        'APP': 'APPSEV'
    }
    for col, sev_col in conditions.items():
        df[sev_col] = df.apply(lambda row: 8 if row[col] == 0 and pd.isna(row[sev_col]) else row[sev_col], axis=1)
    
    return df

def recode_vitals(df):
    """
    this converts the weight and height values to harmonize units.
    Apply before renaming to UDS format
    """
    mask = df['VSWTUNIT'] == 2
    # convert weight to pounds for rows where 'VSWTUNIT' is equal to 2
    df.loc[mask, 'VSWEIGHT'] *= 2.20462
    # height
    mask = df['VSHTUNIT'] == 2
    # Convert 'VSHEIGHT' to inches for rows where 'VSHTUNIT' is equal to 2
    df.loc[mask, 'VSHEIGHT'] *= 0.393701
    return df

def recode_faq(df):
    """
    this function tries to match the coding of FAS/FAQ (same thing) in ADNI to the UDS coding.
    Apply this before renaming variables to uds.
    Pass in te faq dataframe
    """
    columns_to_recode = ['FAQFINAN', 'FAQFORM', 'FAQSHOP', 'FAQGAME', 'FAQBEVG', 'FAQMEAL', 'FAQEVENT', 'FAQTV', 'FAQREM', 'FAQTRAVL', 'FAQTOTAL']

    # Define the mapping of old values to new values
    recode_mapping = {1: 0, 2: 1, 3: 1, 4: 2, 5: 3}

    # Recode the values in the specified columns
    for column in columns_to_recode:
        df[column] = df[column].replace(recode_mapping)

    for column in columns_to_recode:
        df[column] = df[column].replace(-1, np.nan)
    return df


def mmse(df):
    # Orientation to time and place
    MMSEORDA = ['MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY', 'MMSEASON']
    MMSEORLO = ['MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE']

    df['MMSEORDA'] =  df[MMSEORDA].eq(1).sum(axis=1)
    df['MMSEORLO'] = df[MMSEORLO].eq(1).sum(axis=1)
    return df


# For smoking hx:
def extract_years(text):
    # Search for a number followed by "years" or "yrs"
    match = re.search(r'(\d+)\s*(years|yrs)', text.lower())
    if match:
        return int(match.group(1))
    return pd.NA
    
def smoking_hx(df):
    """
    Create TOBAC100 based on IHDESC containing "smoker" or "smoked"
    if IHDESC contains "smoker" or "smoked", TOBAC100 is 1, otherwise 0 - this means that the person smoked in their lifetime
    """
    df['TOBAC100'] = df['IHDESC'].apply(lambda x: 1 if any(keyword in x.lower() for keyword in ['smoker', 'smoked']) else 0)

    # Create TOBAC30 based on IHPRESENT and TOBAC100
    # if IHPRESENT is 1, and the columns IHDESC contains "smoker" or "smoked" (equivalent to TOBAC100 = 1), then TOBAC30 is 1. Else 0
    df['TOBAC30'] = df.apply(lambda x: 1 if x['IHPRESENT'] == 1 and x['TOBAC100'] == 1 else 0, axis=1)
    # apply only where TOBAC100 is 1 - extract years from IHDESC and create SMOKYRS
    df['SMOKYRS'] = df.apply(lambda x: extract_years(x['IHDESC']) if x['TOBAC100'] == 1 else pd.NA, axis=1)


def recode_race(df):
    # apply after renaming demographics to UDS
    recode_dict = {
    'HISPANIC': {2:0},
    'RACE': {1:3, 2:5, 3:4, 4:2, 5:1, 7:np.nan},
    'RESIDENC': {2:1, 3:1, 5:2, 6:3, 7:4, 4:9, 8:9},
    'MARISTAT': {4:5, 5:9},
    'HANDED': {1:2, 2:1}}
    df = df.replace(recode_dict)

    return df