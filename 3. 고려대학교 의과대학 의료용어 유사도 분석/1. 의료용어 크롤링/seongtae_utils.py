def file_finder(directory, ext="txt"): # find every files in a directory (with searching subdirectories)
    import os
    
    pathlist = []
    filelist = []
    folders = [directory] # folders to check
    complete_folders = []
    
    while len(folders) != 0:
        folders_to_add = []
        folders_to_delete = []
        found_entity = False
        
        for folder in folders:
            for entity in os.listdir(folder):
                if os.path.isfile(os.path.join(folder, entity)):
                    if entity[-len(ext):] == ext:
                        pathlist.append(folder+"/"+entity)
                        filelist.append(entity)
                        found_entity = True
                else:
                    folders_to_add.append(folder+"/" + entity)
            #print()
            if found_entity:
                folders_to_delete.append(folder)

        if len(folders_to_add) != 0 or len(folders_to_delete) != 0:
            for folder in folders:
                complete_folders.append(folder)
            folders.clear()
            for folder in folders_to_add:
                folders.append(folder)
            for folder in folders_to_delete:
                if folder in folders:
                    folders.remove(folder)
    return pathlist, filelist

def dic_update(keys_values, dic, typ="list"):
    if typ == "list":
        for key, value in keys_values:
            if key in dic:
                dic[key].append(value)
            else:
                dic[key] = [value]
    elif typ == "dic":
        for key, value in keys_values:
            if key in dic:
                dic[key].update(value)
            else:
                dic[key] = value
    elif typ == "set":
        for key, value in keys_values:
            if key in dic:
                dic[key].add(value)
            else:
                dic[key] = {value}
    return dic
    

class object_saveloader():
    def __init__(self, savename="", loadname="", tp="pkl", interval=100, savepath="./", loadpath="./", vervose=False):
        import os
        assert tp == "pkl" or tp == "json"
        assert savename != "" or loadname != ""
        assert os.path.exists(savepath) and os.path.exists(loadpath)

        self.type=tp
        self.index=0
        self.interval=interval
        self.savepath = savepath
        self.loadpath = loadpath
        self.lastcheckpoint=0
        if savename != "":
            self.savename = savename
        if loadname != "":
            self.loadname = loadname
            
    def index_update(self, var, length):
        assert isinstance(length, int)
        self.index+=1
        if self.index % self.interval == 0 or abs(self.lastcheckpoint-length) >= self.interval:    
            self.save(var)
            self.lastcheckpoint=length
    
    def save(self, var):
        if self.type == "pkl":
            import pickle
            with open(self.savepath+"/"+self.savename+".pkl", "wb") as f:
                pickle.dump(var, f)
        elif self.type == "json":
            import json
            with open(self.savepath+"/"+self.savename+".json", "w", encoding="UTF-8-sig") as f:
                json.dump(var, f)
    
    def load(self, var):
        obj = None
        if self.type == "pkl":
            import pickle
            obj =  pickle.load(open(self.loadpath+"/"+self.loadname+".pkl", "rb"))
        elif self.type == "json":
            import json
            obj =  json.load(open(self.loadpath+"/"+self.loadname+".json", "r", encoding="UTF-8-sig"))
        assert obj is not None
        self.lastcheckpoint = len(obj)
        self.index=len(obj)
        return obj

