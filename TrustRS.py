import PySimpleGUI as sg
import os.path
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from math import sqrt

# sg.theme('LightGreen6')
sg.theme('LightBlue')


def make_window1():
    file_list_column = [
        [sg.Text("Data Folder"), ],
        [

            sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
            sg.FolderBrowse(),
        ],
        [
            sg.Listbox(
                values=[], enable_events=True, size=(34, 20), key="-FILE LIST-"
            )
        ],
    ]

    # For now will only show the name of the file that was chosen
    image_viewer_column = [
        [sg.Text("Choose the data file from list on left:", key="-LABL-")],
        [sg.Text(size=(40, 1), key="-TOUT-")],
        [sg.ProgressBar(1, orientation='h', size=(40, 20), key='progress', visible=False,
                        bar_color=("#6da34d", "#fff"))],
        [sg.Button("Set Review table", key="-SET-RATINGS-TBL-"), sg.Button("Set Trust table", key="-SET-TRUST-TBL-")],
        [sg.Button("Set Similarity table", key="-SET-SIM-TBL-")],
        [sg.Button("Preprocess Data", key="-PREPROCESS-"), sg.Text("Holdout:"),
         sg.InputText("0.2", size=(6, 1), key="-HOLDOUT-")],
        [sg.Radio('Approach 1', "RADIO1", default=True, key="-APP1-"), sg.Text("Beta:"),
         sg.InputText("0.5", size=(6, 1), key="-BETA-")],
        [sg.Radio('Approach 2', "RADIO1", default=False, key="-APP2-")],
        [sg.Button("Compute Trust values", key="-CTRST-")],
        [sg.Button("Predict Ratings", key="-PREDICT-")],
        [sg.Button('Evaluate System', key="-ANALYZE-")]
    ]

    # ----- Full layout -----
    layout = [
        [
            sg.Column(file_list_column),
            sg.VSeperator(),
            sg.Column(image_viewer_column),
        ]
    ]

    return sg.Window("Trust-based Recommender System", layout, font='Default 14', finalize=True)


def make_window2():
    layout = [[sg.Text(size=(40, 1), key="-WIN2-")],
              [sg.Text(size=(40, 3), key='-OUTPUT-')],
              [sg.HorizontalSeparator()],
              [sg.Text("Compare recommendations for User ID:"), sg.InputText(size=(6, 1), key="-USERID-")],
              [sg.Text("Note: User id should be from the test set", text_color="#777777", font='Default 12')],
              [sg.Button("Compare", key="-COMPARE-")],
              [sg.Text("Recommendations based on")],
              [sg.Text("Real ratings:"),
               sg.Text("                                                                       Predicted ratings:")],
              [
                  sg.Table(values=[],
                           headings=['User ID','Product ID','Rating'],
                           display_row_numbers=True,
                           auto_size_columns=False, key="-RLIST-"),
                  sg.Table(values=[],
                           headings=['User ID','Product ID','Predicted Rating'],
                           display_row_numbers=True,
                           auto_size_columns=False, key="-PLIST-")
              ],
              [sg.Button('Back', key="Exit")],
              ]
    return sg.Window('System Evaluation', layout, font='Default 14', finalize=True)


progress_bar = None
Review_tbl = pd.DataFrame()
Trust_tbl = pd.DataFrame()
Sim_tbl = pd.DataFrame()
B = 0
holdout = 0
trust1 = 0
trust2 = 0
train_set = pd.DataFrame()
test_set = pd.DataFrame()
pre_calculations = pd.DataFrame()
ratings_matrix = csr_matrix(([], ([], [])), shape=(0, 0))
cos_similarities = csr_matrix(([], ([], [])), shape=(0, 0))
listOfTrustRelevancies_df = pd.DataFrame()
tAll_file = pd.DataFrame()
predicted = 0

# Uncomment to directly evaluate:
# progress_bar = None
# Review_tbl = pd.read_csv("./" + "Review_sample.csv")  # pd.DataFrame()
# Trust_tbl = pd.read_csv("./" + "Trust_sample.csv")  # pd.DataFrame()
# Sim_tbl = pd.read_csv("./" + "Similarity_sample.csv")  # pd.DataFrame()
# trust1 = 1
# train_set = pd.read_csv("./" + "test_set_example.csv")  # pd.DataFrame()
# test_set = pd.read_csv("./" + "train_set_example.csv")  # pd.DataFrame()
# tAll_file = pd.read_csv("./" + "trust_values_1.csv")  # pd.DataFrame()
# predicted = 1


def computeMultipleRatingsSameUserItem():
    # If one user rated the same item many times then we take the mean of his rating to that item
    agg_Review_tbl = (Review_tbl.groupby(['iduser', 'idproduct']).rating.agg(np.mean).reset_index())
    return agg_Review_tbl


def splitData(data, holdout_fraction=0.2):
    # Splits a DataFrame into training and test sets.
    # holdout_fraction: fraction of data rows to use in the test set.
    test_set = data.sample(frac=holdout_fraction, replace=False)
    train_set = data[~data.index.isin(test_set.index)]
    return train_set, test_set


def computeAvgRatingAndAdjustedRating(data):
    Mean = data.groupby(['iduser'], as_index=False, sort=False).mean().rename(columns={'rating': 'rating_avg'})[
        ['iduser', 'rating_avg']]
    Ratings = pd.merge(data, Mean, on='iduser', how='left', sort=False)
    Ratings['rating_adjusted'] = Ratings['rating'] - Ratings['rating_avg']
    Ratings[['iduser', 'idproduct', 'rating', 'rating_avg', 'rating_adjusted']]
    return Ratings


def computeCosineSimilarity(data):
    pivot_user_based = pd.pivot_table(data, index='iduser', columns=['idproduct'], values='rating')
    user_sparse_pivot = csr_matrix(pivot_user_based.fillna(0))
    similarities_sparse = cosine_similarity(user_sparse_pivot, dense_output=False)
    return similarities_sparse


def computeDegreeCentrality(user_id):
    degree = Trust_tbl.loc[Trust_tbl['idtrusted'] == user_id, :].count().iduser
    return degree


def computeTrustRelevancy(user_i, user_k, e):
    # To calculate the sum of similarities between user i and all users that he trust in the trust list
    trust_user = Trust_tbl.loc[Trust_tbl['iduser'] == user_i, :]
    sumofUserIandUserK_CosSimilarities = 0
    pearson_sim = 0
    val1 = 0
    val2 = 0
    trust = 0

    for index, row in trust_user.iterrows():
        user_trusted = row['idtrusted']
        if user_i > 307738:
            continue
        if user_trusted > 307738:
            continue
        if user_k > 307738:
            continue
            # To check if any value exist and it is not null
        if user_trusted and user_i < cos_similarities.shape[0] and user_trusted < cos_similarities.shape[0]:
            sumofUserIandUserK_CosSimilarities = sumofUserIandUserK_CosSimilarities + cos_similarities[
                user_i, user_trusted]

    if user_i < 307738 and user_trusted < 307738 and user_k < 307738:
        pearson_sim = Sim_tbl.loc[(Sim_tbl['iduser'] == user_i) & (Sim_tbl['idsimilar'] == user_k), 'similarity']
        if pearson_sim.size > 0:
            pearson_sim = pearson_sim.values[0]
        else:
            pearson_sim = 0

    if e == 1:
        if user_i < cos_similarities.shape[0] and user_k < cos_similarities.shape[0]:
            val1 = (B * (((cos_similarities[user_i, user_k]) / (sumofUserIandUserK_CosSimilarities)) + pearson_sim))
    elif e == 2:
        val1 = pearson_sim

    # To calculate the sum of the degree centralities for all trusted users who are trusted by user i in the trust list
    sumofTrustedUsers_DegreeCentralities = 0
    for index, row in trust_user.iterrows():
        user_trusted = row['idtrusted']
        # To check if any value exist and it is not null
        if user_trusted and user_i < 307738 and user_trusted < 307738 and user_k < 307738:
            sumofTrustedUsers_DegreeCentralities = sumofTrustedUsers_DegreeCentralities + computeDegreeCentrality(
                user_trusted)

    if e == 1:
        val2 = ((1 - B) * ((computeDegreeCentrality(user_k)) / (sumofTrustedUsers_DegreeCentralities)))
    elif e == 2:
        val2 = ((computeDegreeCentrality(user_k)) / sumofTrustedUsers_DegreeCentralities)

    if e == 1:
        trust = val1 + val2
    elif e == 2:
        trust = val1 * val2

    return trust


def preComputeAllTrustsforUser_new(user):
    trust_user = tAll_file.loc[tAll_file['iduser'] == user, :][['iduser', 'idtrusted', 'trust_relevancy']]
    return trust_user


def computeAllTrustsforAllUsers(e):
    all_trust_users = Trust_tbl[['iduser', 'idtrusted']]
    trust_relevancies = Trust_tbl.rename(columns={'trust': 'trust_relevancy'})[
        ['iduser', 'idtrusted', 'trust_relevancy']]
    if len(all_trust_users.index) > 0:
        for i in range(0, len(all_trust_users)):
            row = all_trust_users.iloc[i]
            user = row['iduser']
            user_trusted = row['idtrusted']
            # check if any value exist and it is not null
            if user_trusted and user < 307738 and user_trusted < 307738:
                trust_relevancies.loc[i, 'trust_relevancy':'trust_relevancy'] = computeTrustRelevancy(user,
                                                                                                      user_trusted, e)
            progress_bar.UpdateBar(i + 1, len(all_trust_users))
        listOfTrustRelevancies_df = pd.merge(all_trust_users, trust_relevancies, on=['iduser', 'idtrusted'],
                                             how='left', sort=False)
        listOfTrustRelevancies_df.to_csv('./trust_values_' + str(e) + '.csv', index=False)
    else:
        return 0


def predictRating_test(user, item):
    # to predict the ratings of user i for the provided data
    # Avg Rating user
    avg_r_user_i = pre_calculations.loc[(pre_calculations['iduser'] == user), 'rating_avg']
    if avg_r_user_i.size > 0:
        avg_r_user_i = avg_r_user_i.values[0]
    else:
        avg_r_user_i = 0
    tAll = preComputeAllTrustsforUser_new(user)
    if len(tAll) > 0:
        sumNumerator = 0
        sumDenominator = 0
        for r in range(len(tAll)):
            numerator = 0
            row = tAll.iloc[r]
            # Adjusted Rating user k item i
            adj_rating = pre_calculations.loc[(pre_calculations['iduser'] == row['idtrusted']) & (
                    pre_calculations['idproduct'] == item), 'rating_adjusted']
            if adj_rating.size > 0:
                adj_rating = adj_rating.values[0]
                tuk = row['trust_relevancy']
                numerator = tuk * adj_rating
                sumNumerator = sumNumerator + numerator
                sumDenominator = sumDenominator + tuk
        if sumDenominator > 0:
            p = avg_r_user_i + (sumNumerator / sumDenominator)
        else:
            p = avg_r_user_i
        return p
    else:
        return 0 + avg_r_user_i


def predictRatingTCF_all(e):
    test_set_zeros = pd.read_csv("./test_set_example.csv")
    test_set_zeros["rating"].apply(lambda x: 0)
    test_set_zeros.to_csv('./zerooz.csv', index=False)
    all_test = test_set_zeros[['iduser', 'idproduct']]
    predicted_rating_TCF = test_set_zeros.rename(columns={'rating': 'predicted_rating'})[
        ['iduser', 'idproduct', 'predicted_rating']]
    if len(all_test.index) > 0:
        for i in range(0, len(all_test)):
            row = all_test.iloc[i]
            user = row['iduser']
            product = row['idproduct']
            predicted_rating_TCF.loc[i, 'predicted_rating':'predicted_rating'] = predictRating_test(user, product)
            progress_bar.UpdateBar(i + 1, len(all_test))
        TCF_df = pd.merge(all_test, predicted_rating_TCF, on=['iduser', 'idproduct'], how='left', sort=False)
        TCF_df.to_csv('./predicted_ratings_' + str(e) + '.csv', index=False)
    else:
        return 0


def evaluate(e):
    real_file = pd.read_csv("./" + "test_set_example.csv")
    prediction_file = pd.read_csv("./" + "predicted_ratings_" + str(e) + ".csv")
    p_col = prediction_file[['predicted_rating']]
    r_col = real_file[['rating']]
    mae = mean_absolute_error(r_col, p_col)
    mse = mean_squared_error(r_col, p_col)
    rmse = sqrt(mean_squared_error(r_col, p_col))
    results = [mae, mse, rmse]
    return results


def getRecommandations_real(uid):
    real_file = pd.read_csv("./" + "test_set_example.csv")
    sortedRecR = real_file.loc[real_file['iduser'] == uid, :]
    sortedRecR.sort_values(by=['rating'], ascending=False)
    header_list = sortedRecR.iloc[0].tolist()
    # Drops the first row in the table (otherwise the header names and the first row will be the same)
    sortedRec = sortedRecR.head(11)
    rlist = sortedRec[1:].values.tolist()
    return header_list, rlist


def getRecommandations_pred(e, uid):
    prediction_file = pd.read_csv("./" + "predicted_ratings_" + str(e) + ".csv")
    sortedRecP = prediction_file.loc[prediction_file['iduser'] == uid, :]
    sortedRecP.sort_values(by=['predicted_rating'], ascending=False)
    header_list = sortedRecP.iloc[0].tolist()
    # Drops the first row in the table (otherwise the header names and the first row will be the same)
    sortedRec = sortedRecP.head(11)
    plist = sortedRec[1:].values.tolist()
    return header_list, plist


def savefiles():
    train_set.to_csv('./train_set_example.csv', index=False)
    test_set.to_csv('./test_set_example.csv', index=False)


window2 = None
window1 = make_window1()
progress_bar = window1.FindElement('progress')
filename = ""

# Run the Event Loop
while True:
    window, event, values = sg.read_all_windows()
    if event == sg.WIN_CLOSED and window == window1:
        break

    if event == "-ANALYZE-" and not window2:
        window1.hide()
        window2 = make_window2()
        if predicted == 0:
            window2['-OUTPUT-'].update("Experiment should be done first!")
            window2["-OUTPUT-"].Update(text_color='#a31621')
        else:
            if values["-APP1-"] == True:
                window2['-WIN2-'].update("The evaluation measures results of Approach 1:")
                window2["-WIN2-"].Update(text_color='#000')
                results = evaluate(1)
            else:
                window2['-WIN2-'].update("The evaluation measures results of Approach 2:")
                window2["-WIN2-"].Update(text_color='#000')
                results = evaluate(2)
            window2['-OUTPUT-'].update(
                "MAE: " + "%.5f" % results[0] + "\nMSE: " + "%.5f" % results[1] + "\nRMSE: " + "%.5f" % results[2])
    if event == "-COMPARE-":
        if predicted == 0:
            window2['-OUTPUT-'].update("Experiment should be done first!")
            window2["-OUTPUT-"].Update(text_color='#a31621')
        else:
            if predicted == 1:
                window2['-WIN2-'].update("The evaluation results of Approach 1:")
                window2["-WIN2-"].Update(text_color='black')
                uid = int(values['-USERID-'])
                if uid and uid != 0:
                    header_listr, rlist = getRecommandations_real(uid)
                    header_listp, plist = getRecommandations_pred(1, uid)
                    window2['-RLIST-'].update(values=rlist,
                                              num_rows=min(25, len(rlist)))
                    window2['-PLIST-'].update(values=plist,
                                              num_rows=min(25, len(plist)))
            else:
                window2['-WIN2-'].update("The evaluation results of Approach 2:")
                window2["-WIN2-"].Update(text_color='black')
                uid = int(values['-USERID-'])
                if uid and uid != 0:
                    header_listr, rlist = getRecommandations_real(uid)
                    header_listp, plist = getRecommandations_pred(2, uid)
                    window2['-RLIST-'].update(values=rlist,
                                              headings=header_listr,
                                              num_rows=min(25, len(rlist)))
                    window2['-PLIST-'].update(values=plist,
                                              headings=header_listr,
                                              num_rows=min(25, len(plist)))

    if window == window2 and (event in (sg.WIN_CLOSED, 'Exit')):
        window2.close()
        window2 = None
        window1.un_hide()
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = os.listdir(folder)
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
               and f.lower().endswith((".csv"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        try:
            filename = os.path.join(
                values["-FOLDER-"], values["-FILE LIST-"][0]
            )
            window["-TOUT-"].update(filename)

        except:
            pass
    elif event == "-SET-RATINGS-TBL-":  # Set Review table
        if filename:
            Review_tbl = pd.read_csv(filename)
            window["-TOUT-"].update("Review table is imported successfully!")
            window["-TOUT-"].Update(text_color='#228F50')  # Green

        else:
            window["-TOUT-"].update("A Review table should be selected!")
            window["-TOUT-"].Update(text_color='#a31621')
    elif event == "-SET-TRUST-TBL-":  # Set Trust table
        if filename:
            Trust_tbl = pd.read_csv(filename)
            window["-TOUT-"].update("Trust table is imported successfully!")
            window["-TOUT-"].Update(text_color='#228F50')  # Green
        else:
            window["-TOUT-"].update("A Trust table should be selected!")
            window["-TOUT-"].Update(text_color='#a31621')
    elif event == "-SET-SIM-TBL-":  # Set Similarity table
        if filename:
            Sim_tbl = pd.read_csv(filename)
            window["-TOUT-"].update("Similarity table is imported successfully!")
            window["-TOUT-"].Update(text_color='#228F50')  # Green
        else:
            window["-TOUT-"].update("A Similarity table should be selected!")
            window["-TOUT-"].Update(text_color='#a31621')
    elif event == "-SET-TRST-VALS-":  # Set already computed trust
        if filename:
            tAll_file = pd.read_csv(filename)
            trust1 = 1
            trust2 = 1
            window["-TOUT-"].update("Trust values is imported successfully!")
            window["-TOUT-"].Update(text_color='#228F50')  # Green
        else:
            window["-TOUT-"].update("A Trust values file should be selected!")
            window["-TOUT-"].Update(text_color='#a31621')
    elif event == "-PREPROCESS-":
        if Review_tbl.empty or Trust_tbl.empty or Sim_tbl.empty:
            window["-TOUT-"].update("Tables should be selected!")
            window["-TOUT-"].Update(text_color='#a31621')
        else:
            # Preprocessing: Data splitting
            data = computeMultipleRatingsSameUserItem()
            holdout = float(values['-HOLDOUT-'])
            if holdout and holdout != 0:
                train_set, test_set = splitData(data, holdout_fraction=holdout)
            else:
                train_set, test_set = splitData(data, holdout_fraction=0.2)
            savefiles()
            # Collaborative filtering model
            pre_calculations = computeAvgRatingAndAdjustedRating(train_set)
            # ratings_matrix = createSparseRatingMatrix(train_set)
            cos_similarities = computeCosineSimilarity(train_set)
            window["-TOUT-"].update("Preprocessing is completed!")
            window["-TOUT-"].Update(text_color='#228F50')  # Green
    elif event == "-CTRST-":
        if values["-APP1-"] == True:
            if Review_tbl.empty or Trust_tbl.empty or Sim_tbl.empty:
                window["-TOUT-"].update("Tables should be selected first!")
                window["-TOUT-"].Update(text_color='#a31621')
            elif pre_calculations.empty or cos_similarities.nnz == 0:
                window["-TOUT-"].update("Preprocessing should be done first!")
                window["-TOUT-"].Update(text_color='#a31621')
            else:
                window["-TOUT-"].update("Loading ...")
                window["progress"].Update(current_count=0, visible=True)
                beta = float(values['-BETA-'])
                if beta and beta != 0:
                    B = beta
                computeAllTrustsforAllUsers(1)
                trust1 = 1
                window["progress"].Update(current_count=0, visible=False)
                window["-TOUT-"].update("Trust values are computed successfully!")
                window["-TOUT-"].Update(text_color='#228F50')  # Green
        elif values["-APP2-"] == True:
            if Review_tbl.empty or Trust_tbl.empty or Sim_tbl.empty:
                window["-TOUT-"].update("Tables should be selected first!")
                window["-TOUT-"].Update(text_color='#a31621')
            elif pre_calculations.empty or cos_similarities.nnz == 0:
                window["-TOUT-"].update("Preprocessing should be done first!")
                window["-TOUT-"].Update(text_color='#a31621')
            else:
                window["-TOUT-"].update("Loading ...")
                window["progress"].Update(current_count=0, visible=True)
                computeAllTrustsforAllUsers(2)
                trust2 = 1
                window["progress"].Update(current_count=0, visible=False)
                window["-TOUT-"].update("Trust values are computed successfully!")
                window["-TOUT-"].Update(text_color='#228F50')  # Green
    elif event == "-PREDICT-" and values["-APP1-"] == True:
        if Review_tbl.empty or Trust_tbl.empty or Sim_tbl.empty:
            window["-TOUT-"].update("Tables should be selected first!")
            window["-TOUT-"].Update(text_color='#a31621')
        elif pre_calculations.empty or cos_similarities.nnz == 0:
            window["-TOUT-"].update("Preprocessing should be done first!")
            window["-TOUT-"].Update(text_color='#a31621')
        elif trust1 == 0:
            window["-TOUT-"].update("Trust values of experiment 1 should be computed first!")
            window["-TOUT-"].Update(text_color='#a31621')
        else:
            tAll_file = pd.read_csv("./" + "trust_values_1.csv")
            window["-TOUT-"].update("Loading ...")
            window["progress"].Update(current_count=0, visible=True)
            predictRatingTCF_all(1)
            predicted = 1
            window["progress"].Update(current_count=0, visible=False)
            window["-TOUT-"].update("Ratings prediction is completed successfully!")
            window["-TOUT-"].Update(text_color='#228F50')  # Green
    elif event == "-PREDICT-" and values["-APP2-"] == True:
        if Review_tbl.empty or Trust_tbl.empty or Sim_tbl.empty:
            window["-TOUT-"].update("Tables should be selected first!")
            window["-TOUT-"].Update(text_color='#a31621')
        elif pre_calculations.empty or cos_similarities.nnz == 0:
            window["-TOUT-"].update("Preprocessing should be done first!")
            window["-TOUT-"].Update(text_color='#a31621')
        elif trust2 == 0:
            window["-TOUT-"].update("Trust values of experiment 2 should be computed first!")
            window["-TOUT-"].Update(text_color='#a31621')
        else:
            tAll_file = pd.read_csv("./" + "trust_values_2.csv")
            window["-TOUT-"].update("Loading ...")
            window["progress"].Update(current_count=0, visible=True)
            predictRatingTCF_all(2)
            predicted = 2
            window["progress"].Update(current_count=0, visible=False)
            window["-TOUT-"].update("Ratings prediction is completed successfully!")
            window["-TOUT-"].Update(text_color='#228F50')  # Green
window1.close()
