# This library is there to handle the complicated config file handling. The check_compartibility
# function is to check in indeed the new dictionary mergeabe with the old one. It should be a part
# of the Git CI. merge_dict will take in the new dictionary and make the associated changes to a
# copy of the main dictionary.


# Smoke test
def check_compartibility(master,slave):
    for key in slave.keys():
        if key in master.keys():
            if type(slave[key]) == type(master[key]):
                if type(slave[key]) == dict:
                    code = check_compartibility(master[key],slave[key])
                    if code == -1:
                        return -1
            else:
                print(f"Type Mismatch, expected value of KEY: {key} to be same in {type(slave[key])} (SLAVE) and {type(master[key])} (MASTER)")
                return -1
        else:
            print(f"Key Mismatch, {key} does not exist")
            return -1
    return 1

def merge_dict(master,slave):
    # The values of the master dictionary will be replaced into a new dictionary called daughter that inherits the changed values of slave
    daughter = master.copy()
    for key in slave.keys():
        if type(slave[key]) != dict:
            daughter[key] = slave[key]
        else:
            daughter[key] = merge_dict(master[key],slave[key])
    return daughter