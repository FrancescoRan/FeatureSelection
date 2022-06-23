"""
This file contains general purpose utilities for all src modules and submodules
"""

from configparser import ConfigParser
import os
import yaml
import csv
import pandas as pd
import re
import cx_Oracle

from typing import Union

def _get_delimiter(file_path, bytes=4096) -> str:
        """
        :param file_path: str
            files' path
        :param bytes: int
            default 4096
        :return: str
            delimiter
        """
        sniffer = csv.Sniffer()
        data = open(file_path, "r").read(bytes)
        delimiter = sniffer.sniff(data).delimiter

        return delimiter

class Utils:
    def __init__(self) -> None:
        pass

    def read_yaml(file_path, arrays : bool = False):
        """
        given filepath reads a yaml file and returns it as python dict
        """
        if arrays:
            with open(file_path, "r") as stream:
                my_yaml = yaml.load(stream, Loader=yaml.UnsafeLoader)
        
        else:
            with open(file_path, "r") as stream:
                my_yaml = yaml.load(stream, Loader=yaml.FullLoader)
        
        return my_yaml


    def read_sql_file(file_path):
        """
        reads a sql file and returns it as python string
        """
        return open(file_path, mode="r", encoding="utf-8-sig").read()

    def write_sql_file(path, query):
        """
        takes a python string and writes a sql file in a certain path. If the file exist, it gets overwritten.
        """
        try:
            with open(path, "x") as q:
                q.write(query)
        except:
            with open(path, "w") as q:
                q.write(query)

    def get_sql_files_directory(self, mydir):
        """
        collects the sql files in a given folder and returns a nested dict of the following format {'sql_filename': {'filepath': '/dummy.sql', 'params': {}}}
        """
        my_dict = {}
        for file in os.listdir(mydir):
            if file.endswith(".sql"):

                # print(file, type(file))

                query_dict = {"filepath": os.path.join(mydir, file)}

                if os.path.isfile(os.path.join(mydir, file.replace(".sql", ".yaml"))):
                    query_dict["params"] = self.read_yaml(
                        os.path.join(mydir, file.replace(".sql", ".yaml"))
                    )

                my_dict[file] = query_dict

        return my_dict
 
    

    def read_file_from_memory(
        self,
        file_path: str,
        file_type: str,
        nrows: Union[int, None] = None,
        columns_to_select: Union[list, None] = None,
        delimiter: Union[str,None] = None
    ) -> pd.DataFrame:
        """
        :param file_name: str
            file name with format
        :param file_type: str
            either sas or csv supported
        :return: pd.DataFrame
            returns the read dataframe
        """

        assert file_type in [
            "sas",
            "csv",
        ], "The only supported files are 'sas' or 'csv', provided: {}".format(file_type)

        if file_type == "sas":
            df = pd.read_sas(
                file_path, encoding="ISO-8859-1", chunksize=None
            )  # in order to avoid bytes in sas files import
 
        elif file_type == "csv":
            if not delimiter:
                delimiter = _get_delimiter(file_path)
            df = pd.read_csv(
                file_path, sep=delimiter, nrows=nrows, usecols=columns_to_select
            )

        print("Replacing '\s' with '_' in columns' names")
        df.columns = [
            re.sub("\s+", "_", c) for c in df
        ]  # replace spaces in columns with "_"

        return df

    def read_table_from_oracle(
        table_name, con, columns_to_select="*", limit=""
    ) -> pd.DataFrame:

        if isinstance(columns_to_select, list):

            columns_to_select = ",".join(columns_to_select)

        query = "select {columns_to_select} from {table_name}".format(
            columns_to_select=columns_to_select, table_name=table_name
        )

        if limit:

            query = query + " where rownum <= {}".format(limit)

        df = pd.read_sql(query, con)

        return df

    def get_connection_curs(self, conf):

        print("CONNECTING TO ORACLE")

        try:
            con = cx_Oracle.connect(
                **self.get_lpd_database_conf(conf["CREDENTIALS"]["LOCAL"])
            )

        except:
            conf_cdsw = conf["CREDENTIALS"]["CDSW"]
            dsn_param = {k: v for k, v in conf_cdsw.items() if k in ["host", "port", "sid"]}
            con_param = {k: v for k, v in conf_cdsw.items() if k in ["user", "password"]}

            dsn = cx_Oracle.makedsn(**self.get_lpd_database_conf(dsn_param))

            con_param.update({"dsn": dsn})
            con = cx_Oracle.connect(**con_param)

        curs = con.cursor()
        print("Database version:", con.version)
        print("Client version:", cx_Oracle.clientversion())

        return con, curs

    def load_data(
        self,
        name_input_tab : str,
        in_memory_folder_input : str,
        from_memory: bool,
        conf: Union[dict, bool] = False,
        limit: Union[int, None] = None,
        columns_to_select: Union[list, None] = None,
    ) -> pd.DataFrame:
        
        file_type = name_input_tab.split('.')[1]
        
        if from_memory:

            # file_name = os.path.join(in_memory_folder_input, name_input_tab, file_type)

            # self.logger.info("LOADING DATA FROM MEMORY: {}".format(file_name))

            # file_path = os.path.join(str(self.folder_path.parent.parent), file_name)

            file_path = os.path.join(in_memory_folder_input, name_input_tab)
            

            df = self.read_file_from_memory(
                file_path=file_path,
                file_type=file_type,
                nrows=limit,
                columns_to_select=columns_to_select
            )
        else:

            # self.logger.info("LOADING DATA FROM ORACLE: {}".format(name_input_tab))
            assert conf, "Please provide a valid conf"
            
            con, curs = self.get_connection_curs(conf)

            df = self.read_table_from_oracle(
                table_name=name_input_tab,
                con=con,
                columns_to_select=columns_to_select,
                limit=limit,
            )

        return df

    def create_folder_if_not_exists(self, folder_path):

        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
            print("Created folder: {}".format(folder_path))
        else:
            pass