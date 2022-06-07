import numpy as np
import pickle








# seed = 0
# nb_final_class = 277

# seed = 1
# nb_final_class = 279

# seed = 2
# nb_final_class = 276

# seed = 3
# nb_final_class = 277

seed = 4
nb_final_class = 276


nb_class = 293
tail_size = [50]

result_path = "/afs/crc.nd.edu/user/j/jhuang24/scratch_50/jhuang24/" \
              "models/msd_net/2022-02-13/known_only_cross_entropy/" \
              "seed_" + str(seed) + "/openmax_feature/"




def calculate_mcc(true_pos,
                  true_neg,
                  false_pos,
                  false_neg):
    """

    :param true_pos:
    :param true_neg:
    :param false_pos:
    :param false_negtive:
    :return:
    """

    return (true_neg*true_pos-false_pos*false_neg)/np.sqrt((true_pos+false_pos)*(true_pos+false_neg)*
                                                           (true_neg+false_pos)*(true_neg+false_neg))



def get_results(known_npy_file,
                unknown_npy_file,
                nb_class,
                label_map,
                nb_final_class):
    """

    :param npy_file:
    :return:
    """
    correct = 0
    wrong = 0

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for one_result in known_npy_file:
        try:
            original_label = one_result[0]

            """
            Map the labels because weibull is missing some classes:
            
            original label: 
                known: 0 ~ 292
                unknown: 294 (293 is known unknown and is not used)
            
            mapped label:
                           
            """

            if original_label != nb_class+1:
                label = label_map[str(original_label).zfill(4)]
            else:
                label = nb_final_class

            openmax_pred = one_result[2]

            # Get the 4 metrics
            if (openmax_pred != nb_final_class):
                true_positive += 1

            if (openmax_pred == nb_final_class):
                false_negative += 1

            # Get the normal accuracy (top-1)
            if openmax_pred == label:
                correct += 1
            else:
                wrong += 1

        except:
            pass

    for one_result in unknown_npy_file:
        openmax_pred = one_result[2]

        if openmax_pred == nb_final_class:
            true_negative += 1
        else:
            false_positive += 1

    print("True positive: ", true_positive)
    print("True negative: ", true_negative)
    print("False postive: ", false_positive)
    print("False negative: ", false_negative)

    accuracy = float(correct)/float(correct+wrong)
    precision = float(true_positive)/float(true_positive+false_positive)
    recall = float(true_positive)/float(true_positive+false_negative)
    f1 = (2*precision*recall)/(precision+recall)
    mcc = calculate_mcc(true_pos=float(true_positive),
                        true_neg=float(true_negative),
                        false_pos=float(false_positive),
                        false_neg=float(false_negative))

    print("Accuracy: ", accuracy)
    print("F-1: ", f1)
    print("MCC: ", mcc)




if __name__ == "__main__":
    label_mapping_path = result_path + "label_mapping.pkl"
    with open(label_mapping_path, 'rb') as f:
        label_mapping = pickle.load(f)

    for one_tail_size in tail_size:

        print("*" * 40)
        print("tail size: ", one_tail_size)

        train_result_path = result_path + "train_results_tail_size_" + str(one_tail_size) + ".npy"
        valid_result_path = result_path + "valid_results_tail_size_" + str(one_tail_size) + ".npy"
        test_known_result_path = result_path + "test_known_results_tail_size_" + str(one_tail_size) + ".npy"
        test_unknown_result_path = result_path + "test_unknown_results_tail_size_" + str(one_tail_size) + ".npy"

        train_result = np.load(train_result_path)
        valid_result = np.load(valid_result_path)
        test_known_result = np.load(test_known_result_path)
        test_unknown_result = np.load(test_unknown_result_path)

        get_results(known_npy_file=test_known_result,
                    unknown_npy_file=test_unknown_result,
                    nb_class=nb_class,
                    label_map=label_mapping,
                    nb_final_class=nb_final_class)
