import json
import requests
import secrets
import pandas as pd
import sqlite3
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import geopandas as gpd
import folium
import webbrowser
import sqlalchemy
import altair as alt
import csv
from pathlib import Path

path = Path.cwd()

client_key = secrets.CENSUS_API_KEY
CACHE_FILE_NAME = "census_cache.json"
CACHE_DICT = {}

def aggregate_data(name, data_frame, column_index):
    '''aggregate data of the same variable

    Parameters
    ---------
    name, dataframe, column index

    Returns
    ----------
    Panda data frame
    '''
    col = int(column_index)
    df = data_frame.iloc[:, 1:col]
    data_frame[name] = df.sum(axis = 1)
    data_frame = data_frame.drop(columns = df)
    return data_frame

def load_cache():
    ''' Opens the cache file if it exists and loads the JSON into
    the CACHE_DICT dictionary.
    if the cache file doesn't exist, creates a new cache dictionary

    Parameters
    ----------
    None

    Returns
    -------
    The opened cache: dict
    '''
    try:
        cache_file = open(CACHE_FILE_NAME, 'r')
        cache_file_contents = cache_file.read()
        cache = json.loads(cache_file_contents)
        cache_file.close()
    except:
        cache = {}
    return cache

def save_cache(cache):
    ''' Saves the current state of the cache to disk

    Parameters
    ----------
    cache_dict: dict
        The dictionary to save

    Returns
    -------
    None
    '''
    cache_file = open(CACHE_FILE_NAME, 'w')
    contents_to_write = json.dumps(cache)
    cache_file.write(contents_to_write)
    cache_file.close()

def construct_unique_key(baseurl, params):
    ''' constructs a key that is guaranteed to uniquely and
    repeatably identify an API request by its baseurl and params


    Parameters
    ----------
    baseurl: string
        The URL for the API endpoint
    params: dict
        A dictionary of param:value pairs

    Returns
    -------
    string
        the unique key as a string
    '''
    alphabetized_keys = sorted(params.keys())
    param_strings = []
    for key in alphabetized_keys:
        param_strings.append("{}_{}".format(key, params[key]))
    return baseurl + "_" + "_".join(param_strings)

def make_request(baseurl, params):
    '''Make a request to the Web API using the baseurl and params

    Parameters
    ----------
    baseurl: string
        The URL for the API endpoint
    params: dictionary
        A dictionary of param:value pairs

    Returns
    -------
    dict
        the data returned from making the request in the form of
        a dictionary
    '''
    new_request = requests.get(baseurl, params).json()
    # print(new_request.headers['content-type'])
    # new_request= new_request.setHeader(charset ='UTF-8')
    # new_request = new_request.json()
    return new_request

def make_request_with_cache(baseurl, get, county, state ):
    '''Check the cache for a saved result for this baseurl+params:values
    combo. If the result is found, return it. Otherwise send a new
    request, save it, then return it.


    Parameters
    ----------
    baseurl: string
        The URL for the API endpoint
    hashtag: string
        The hashtag to search for
    count: integer
        The number of results you request from Twitter

    Returns
    -------
    dict
        the results of the query as a dictionary loaded from cache
        JSON
    '''
    try:
        with open(CACHE_FILE_NAME) as my_data:
            CACHE_DICT = json.load(my_data)
        my_data.close()
    except:
        CACHE_DICT = {}


    params_dict = {}
    params_dict['get'] = get
    params_dict['for'] = county
    params_dict['in'] = state

    uniq_url = construct_unique_key(baseurl, params_dict)
    # print(uniq_url)

    if uniq_url in CACHE_DICT.keys():
        print("\nInitializing...")
        print("Fetching cached data...")
        print("Populating database...")
    else:
        print("\nInitializing...")
        print("Making a new request!")
        print("Populating database...")
        CACHE_DICT[uniq_url] = make_request(baseurl, params_dict)
        print(CACHE_DICT[uniq_url])
        with open(CACHE_FILE_NAME, 'w') as outfile:
            outfile.write(json.dumps(CACHE_DICT, indent=2))
        outfile.close()
    return CACHE_DICT[uniq_url]

def percentileTable(in_data, fields):
    """This function Calculates the percentile score for each
    value in a numeric field.

    Parameters:
        in_data - input pandas dataframe containing values to be
       ranked.
    fields - list containing names of fields within dataframe
       to be ranked.

    Output:
    The ouput is a pandas dataframe containing the newly created
    percentile fields.
    """
    for field in fields:
        vals = list(in_data[field])
        arr = [i for i in vals if i != 0]
        pctile = [stats.percentileofscore(
            arr, n) if n != 0 else 0 for n in vals]
        in_data["P_" + field] = pctile

    p_fields = ["P_" + field for field in fields]

    return pd.DataFrame(in_data, columns=p_fields)

def PCA_kaiser(in_data):
    """This function performs a PCA for an input dataset with multiple
    independent variables. The Kaiser rule is applied to the resulting
    components. The output is a pandas dataframe with all remaining
    components with and eigenvalue over 1.00

    Parameter:
    Pandas data frame

    Output:
    pandas data frame"""

    component_cnt = len(in_data.columns)
    X_scaled = StandardScaler().fit_transform(in_data)
    pca = PCA(component_cnt)
    f = pca.fit(X_scaled)
    t = pca.transform(X_scaled)
    PCA_Components = pd.DataFrame(t)
    keep_components = 0
    for eigval in pca.explained_variance_:
        if eigval > 1:
            keep_components = keep_components + 1
    return pd.DataFrame(PCA_Components.iloc[:, 0:keep_components])

def varimax(Phi, gamma = 1.0, q = 20, tol = 1e-6):
    """This function performs a varimax rotation for a set of PCA components.

    Parameters:
    pandas dataframe containing component scores

    Output:
    Pandas dataframe wiht rotated scores
    """
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(
            dot(Phi.T,asarray(
                Lambda)**3 - (gamma/p) * dot(
                    Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d_old!=0 and d/d_old < 1 + tol: break
    return pd.DataFrame(dot(Phi, R))

def factorLoadings(PCA_table, in_table):
    """This function computes the factor loadings for each component
    in a PCA dataset.

    Parameters:
        PCA_table - dataframe containing PCA scores
        in_table - original input dataframe used to create the PCA scores

    Output:
    Pandas dataframe containing factor loading for each component
    """
    compare = pd.concat([PCA_table, in_table], axis=1, sort=False)
    corr = compare.corr()
    pca_cols = len(PCA_table.columns)
    return corr.iloc[pca_cols:, :pca_cols]



if __name__ == "__main__":

    CACHE_DICT = load_cache()

    HOST = 'https://api.census.gov/data'
    year = '2016'
    dataset = 'acs/acs5'
    baseurl = "/".join([HOST, year, dataset])

    get_vars = [
    'NAME',
    #Female Population
    'B01001_026E',
    #Under Age 10
    "B01001_003E", "B01001_004E", "B01001_027E","B01001_028E",
    ## Over age 64:
    "B01001_020E", "B01001_021E", "B01001_022E",
    "B01001_023E", "B01001_024E", "B01001_025E",
    "B01001_044E", "B01001_045E", "B01001_046E",
    "B01001_047E", "B01001_048E", "B01001_049E",

    # # With a disability:
    "B18101_004E", "B18101_007E", "B18101_010E",
    "B18101_013E", "B18101_016E", "B18101_019E",
    "B18101_023E", "B18101_026E", "B18101_029E",
    "B18101_032E", "B18101_035E", "B18101_038E",

    # # Poverty status:
    "B17020_002E",
    # Unemploymed:
    "C18120_006E",
                    #
    #Part-time workers:
    "B23027_005E", "B23027_010E", "B23027_015E",
    "B23027_020E", "B23027_030E", "B23027_035E",
    #
    # Renters:
    "B25009_010E",
    #
    # Recieve public assistance:
    "B19057_002E",
    #
    # Single parent households:
    "B09005_004E", "B09005_005E",
    #
    # No vehicle available:
    "B08201_002E"
    ]

    #Access Census API
    predicates = {}
    predicates['get'] = ",".join(get_vars)
    get = ",".join(get_vars) #params: list of varaibles
    county = "county:*" #for: geography of interest
    state = "state:72"

    predicates['get'] = get
    predicates['for'] = county
    predicates['in'] = state
    census = make_request(baseurl=baseurl, params= predicates)


    census_data = make_request_with_cache(baseurl, get, county, state)
    header = census_data[0]

    #cleaning accessed data
    df = pd.DataFrame(columns = header, data = census_data[1:])
    df.iloc[:, 1:] = df.iloc[:, 1:].astype(int)
    df = aggregate_data('TotalFemalePopulation', df, 2)
    df = aggregate_data('UnderAge10', df, 5)
    df = aggregate_data('OverAge64', df, 13)
    df = aggregate_data('WithADisability', df, 13)
    df = aggregate_data('PovertyStatus', df, 2)
    df = aggregate_data('Unemployment', df, 2)
    df = aggregate_data('ParttimeWorkers', df, 7)
    df = aggregate_data('Renters', df, 2)
    df = aggregate_data('ReceivePublicAssistance', df, 2)
    df = aggregate_data('SingleParentHouseholds', df, 3)
    df = aggregate_data('NoVehicleAvailable', df, 2)

    #Perform PCA analysis to calculate social vulnerability score
    fields = ['TotalFemalePopulation','UnderAge10','OverAge64','WithADisability','PovertyStatus','Unemployment',
    'ParttimeWorkers','Renters','ReceivePublicAssistance','SingleParentHouseholds','NoVehicleAvailable']
    percentile_df = percentileTable(df, fields)
    in_PCA = PCA_kaiser(percentile_df)
    PCA_rotated = varimax(in_PCA)
    f1 = factorLoadings(PCA_rotated, percentile_df)
    PCA_rotated = PCA_rotated * -1

    percentile_df['SocialVulnerabilityScore'] = PCA_rotated
    percentile_df['Name'] = df['NAME']
    percentile_df['State'] = df['state']
    percentile_df['FIPS_Code'] = df['county']
    percentile_df['Name'] = percentile_df['Name'].str.replace(r' Municipio, Puerto Rico$','')

    #Shape file for mapping
    counties = gpd.read_file(path.joinpath('tl_2016_us_county.shp'), encoding= 'UTF-8')
    pr = counties[counties.STATEFP == '72']

    #Create database for census and agriculture data
    conn = sqlite3.connect('project_data.sqlite')
    percentile_df.to_sql('CENSUS', conn, if_exists = 'replace')

    c = conn.cursor()
    drop_census = '''
        DROP TABLE CENSUS;
    '''

    drop_hub = '''
        DROP TABLE IF EXISTS 'CARRIBEAN_HUB';
    '''

    create_carribean_hub_table = '''
        CREATE TABLE IF NOT EXISTS "CARRIBEAN_HUB"(
            "Id" INTEGER PRIMARY KEY AUTOINCREMENT,
            "Agricultural_Region" TEXT,
            "FIPS_Code" INTEGER,
            "Neighborhood" TEXT,
            "Crop" TEXT,
            'Scientific_Name' TEXT,
            "Production_unit" TEXT,
            "Production"   FLOAT,
            "Production_in_pounds"  FLOAT,
            "Production_in_kilograms"   FLOAT,
            "Planted_cuerdas" FLOAT,
            "Planted_hectares" FLOAT,
            "Planted_square_feet"   FLOAT,
            "Harvested_cuerdas" FLOAT,
            "Harvested_hectares" FLOAT,
            "Harvested_square_feet"  FLOAT
        );
    '''

    c.execute(drop_hub)
    c.execute(create_carribean_hub_table)

    conn.commit()
    conn.close()

    #Insert values into database table
    file_contents = open(path.joinpath('coffee_pr_2016.csv'), 'r', encoding='utf-8')
    csv_reader = csv.reader(file_contents)
    next(csv_reader)

    select_municipality_fips_sql = '''
        SELECT FIPS_CODE FROM CENSUS
        WHERE Name = ?
    '''

    insert_hub_sql = '''
        INSERT INTO CARRIBEAN_HUB
        VALUES(NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''

    conn = sqlite3.connect('project_data.sqlite')
    cur = conn.cursor()

    #Create relational keys
    for row in csv_reader:
        cur.execute(select_municipality_fips_sql, [row[1]])
        res = cur.fetchone()
        municipality_fips = None
        if res is not None:
            municipality_fips = res[0]
        cur.execute(insert_hub_sql, [
            row[0], #Agricultural_Region
            municipality_fips, #municipality_fips()
            row[2],#Neighborhood
            row[3],#Crop
            row[4],#Scientific_Name
            row[5],# Production_unit
            row[6],# Production
            row[7],#Production in pounds
            row[8],#Production in kilograms
            row[9],#Planted cuerdas
            row[10],#Planted hectares
            row[11],#Planted Planted_square_feet
            row[12],# Harvested_cuerdas
            row[13],#harvested hectares
            row[14]#harvested squarefeet
        ])
    conn.commit()
    #Access database to joing to geopandas data. Note: this step could be optimized using PostGIS
    inner_join_query = '''
        SELECT
            Production,
            Production_in_pounds,
            Production_in_kilograms,
            Planted_cuerdas,
            Planted_hectares,
            Planted_square_feet,
            Harvested_square_feet,
            Name

        FROM CARRIBEAN_HUB
            INNER JOIN CENSUS ON CENSUS.FIPS_Code = CARRIBEAN_HUB.FIPS_Code

    '''
    result = cur.execute(inner_join_query).fetchall()
    coffee = pd.read_sql_query(inner_join_query, conn)


    coffee['Production_in_pounds'] = (coffee['Production_in_pounds'].str.replace(',','')).astype(float)
    coffee['Planted_square_feet'] = (coffee['Planted_square_feet'].str.replace(',','').astype(float))
    coffee['Harvested_square_feet'] = (coffee['Harvested_square_feet'].str.replace(',','').astype(float))
    coffee = coffee.groupby('Name', as_index=False).sum()
    coffee['Percentage_Harvested'] = (coffee['Harvested_square_feet'] / coffee['Planted_square_feet'])*100

    socio_economics_var = ['P_TotalFemalePopulation','P_UnderAge10','P_OverAge64','P_WithADisability','P_PovertyStatus','P_Unemployment',
    'P_ParttimeWorkers','P_Renters','P_ReceivePublicAssistance','P_SingleParentHouseholds','P_NoVehicleAvailable', 'SocialVulnerabilityScore']

    #User interactivity
    print("""\nIn 2017, Hurricane Maria struck the United States unincorporated territory of Puerto Rico as a category four hurricane. This project examines some of the socio-economic and agriculture data of Puerto Rico to improve the local smallholder coffee farmer's resilience. The socio-economic data are downloaded from the United States Census Bureau, and the agriculture data are downloaded from the Caribbean Climate Hub. The socio-economic factors are used to create a social vulnerability index to help depict community vulnerability's spatial patterns. Please refer to the document in Github for more information.""")

    print('\nWould you like to view the socio-economics data or agriculture data?')
    print("Press '1' for agriculture data, press '2' for socio-economics data, or'exit' to exit! (Note: the visualization will be saved to your local working directory.)")
    user_response = input('\nEnter response:')
    while user_response != 'exit':
        if user_response == '1':
            print('''\nChoose the number for the visualization you'd like to see or "exit" or "back":\n1. Top Coffee Production Municipalities in Puerto Rico 2016\n2. Coffee Harvested vs. Coffee Planted in 2016 \n3. Percentage coffee harvested per municpality in 2016. ''')
            user_response = input('Enter response:')
            while user_response != 'back':
                if user_response == '1':
                    vis1 = alt.Chart(coffee).transform_window(
                        rank = 'rank(Production_in_pounds)',
                        sort = [alt.SortField('Production_in_pounds', order = 'descending')]
                    ).transform_filter(
                        (alt.datum.rank <20)
                    ).mark_bar().encode(
                        alt.Y('Production_in_pounds:Q', title = 'Coffee Production in pounds'),
                        alt.X('Name:N', title = 'Municipalities', sort = alt.EncodingSortField(
                        field = 'rank', order = 'ascending')
                    )
                    ).properties(
                        width = 800,
                        height = 400
                    ).encode(
                        tooltip = [alt.Tooltip('Production_in_pounds:Q', title = 'Production in pounds', format = ',')]
                    ).properties(
                        title = 'Top Coffee Production Municipalities in Puerto Rico 2016'
                    )
                    vis1.save('vis1.html')
                    webbrowser.open_new_tab(r'C:\python_workfolder\si507\final_project\vis1.html')
                    print('''\nChoose another visualization you'd like to see or "back":\n1. Top Coffee Production Municipalities in Puerto Rico 2016\n2. Coffee Harvested vs. Coffee Planted in 2016 \n3. Percentage coffee harvested per municpality in 2016. ''')
                    user_response = input('Enter response:')
                elif user_response == '2':
                    vis2 = alt.Chart(coffee).mark_circle().encode(
                        alt.X('Planted_square_feet:Q', title = 'Planted coffee (square feet)'),
                        alt.Y('Harvested_square_feet:Q', title = 'Harvested coffee (square feet)'),
                        size = 'Production_in_pounds:Q'
                        ).properties(
                        width = 800,
                        height = 400,
                        title = "Planted Coffee vs. Harvested Coffee In Square Feet in 2016"
                        ).encode(
                        tooltip = [alt.Tooltip('Name:N'), alt.Tooltip('Harvested_square_feet:Q', title = 'Square Feet Harvested',format = ',')]
                        )
                    vis2.save('vis2.html')
                    webbrowser.open_new_tab(r'C:\python_workfolder\si507\final_project\vis2.html')
                    print('''\nChoose another visualization you'd like to see or "exit" or "back":\n1. Top Coffee Production Municipalities in Puerto Rico 2016\n2. Coffee Harvested vs. Coffee Planted in 2016 \n3. Percentage coffee harvested per municpality in 2016. ''')
                    user_response = input('Enter response:')
                elif user_response == '3':
                    vis3 = alt.Chart(coffee).mark_bar().encode(
                        alt.X('Percentage_Harvested:Q', title = 'Percentage coffee harvested'),
                        alt.Y('Name:N')
                        ).properties(
                        width = 800,
                        height = 400,
                        title = 'Percentage Coffee Harvested Per Municiaplity In 2016'
                        ).encode(
                        tooltip = [alt.Tooltip('Percentage_Harvested:Q', title = 'Percentage coffee harvested')]
                        )
                    vis3.save('vis3.html')
                    webbrowser.open_new_tab(r'C:\python_workfolder\si507\final_project\vis3.html')
                    print('''\nChoose another visualization you'd like to see or "exit" or "back":\n1. Top Coffee Production Municipalities in Puerto Rico 2016\n2. Coffee Harvested vs. Coffee Planted in 2016 \n3. Percentage coffee harvested per municpality in 2016. ''')
                    user_response = input('Enter response:')
                else:
                    user_response = input('The input must be the index of the corresponding variable. Please enter a valid option: ')
                    continue
            print('\nWould you like to view the socio-economics data or agriculture data?')
            print("\nPress '1' for agriculture data, press '2' for socio-economics data, or'exit' to exit!")
            user_response = input('\nEnter response:')
        elif user_response == '2':
            print('\n1. Total female population')
            print('2. Under age 10')
            print('3. Over age 64')
            print('4. With a disability')
            print('5. Poverty status')
            print('6. Unemployment')
            print('7. Part time workers')
            print('8. Renters')
            print('9. Receive public assistance')
            print('10. Single parent households')
            print('11. No vehicle available')
            print('12. Social vulnerability index')
            user_response = input('\nPlease enter the index of one of the variables to visualize or enter "back": ')
            while user_response != 'back':
                try:
                    user_response = int(user_response)
                    var = socio_economics_var[user_response-1]
                except (ValueError, TypeError) as e:
                    print('\n[Error! Enter a number, or "back"]')
                    user_response = input('\nPlease enter the index of one of the variables to visualize or enter "back": ')
                    continue
                else:
                    df = pd.read_sql_query("select " + var + ", Name from CENSUS;", conn)
                    table = pr.merge(df, how = "left", left_on = ['NAME'], right_on=['Name'])
                    my_map = folium.Map(location=[18.2208, -66.5901], zoom_start=9)
                    folium.Choropleth(
                        geo_data=table,
                        name='choropleth',
                        data=table,
                        columns=['Name', var],
                        key_on='feature.properties.Name',
                        fill_color='OrRd',
                        fill_opacity=0.7,
                        line_opacity=0.2,
                        legend_name= var
                        ).add_to(my_map)
                    my_map.save(var + '.html')
                    webbrowser.open_new_tab(var + '.html')

                    print('''\nChoose another visualization you'd like to see or "back":\n''')
                    print('1. Total female population')
                    print('2. Under age 10')
                    print('3. Over age 64')
                    print('4. With a disability')
                    print('5. Poverty status')
                    print('6. Unemployment')
                    print('7. Part time workers')
                    print('8. Renters')
                    print('9. Receive public assistance')
                    print('10. Single parent households')
                    print('11. No vehicle available')
                    print('12. Social vulnerability index\n')
                    user_response = input('Enter response:')
                    continue

            print('\nWould you like to view the socio-economics data or agriculture data?')
            print("\nPress '1' for agriculture data, press '2' for socio-economics data, or'exit' to exit!")
            user_response = input('\nEnter response:')

        else:
            user_response = input('Please enter a valid option: ')
            continue
    print('\nBye!')
    conn.close()
