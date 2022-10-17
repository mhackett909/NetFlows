import pandas as pd
import numpy as np
import os

class Extractor:
    def __init__(self, path, file, method):
        print(f"Loading: {file}")
        self.df = pd.read_csv(path+file)
        self.path = path
        self.file = file
        self.method = method
        self.threshold = 2 # Min packets for flow analysis 
        # Ignoring source should improve DDoS detection
        self.id_cols = ['ip.dst', 'dstport', 'ip.proto'] 
        self.feature_cols = ['Pkts_Per_Sec', 'MBits_Per_Sec', 
                             'Pkt_Size_Avg', 'Pkt_Size_Std', 'Pkt_Size_Q1', 'Pkt_Size_Q2', 'Pkt_Size_Q3', 'Pkt_Size_Min', 'Pkt_Size_Max', 
                             #'TCP_Flags_Avg', 'TCP_Flags_Std', 'TCP_Flags_Q1', 'TCP_Flags_Q2', 'TCP_Flags_Q3', 'TCP_Flags_Min', 'TCP_Flags_Max',
                             #'TTL_Avg', 'TTL_Std', 'TTL_Q1', 'TTL_Q2', 'TTL_Q3', 'TTL_Min', 'TTL_Max', 
                             'Anomaly']
    def getSubflowFeatures(self):
        return self.subflow_features
    def getFlowInfo(self):
        # Number of subflows per flow
        flow_num = [] 
        for i in range(len(self.subflow_indices)):
            flow_num.append(len(self.subflow_indices[i]))
        flow_num = pd.Series(flow_num)
        flow_keys = pd.Series(self.keys)
        flow_df = pd.DataFrame([flow_num,flow_keys]).T
        flow_df.columns = ['Num Subflows', 'Flow ID']
        return flow_df
    def dropNaN(self):
        print("Cleaning data...")
        df = self.df
        # NAN values
        df.dropna(subset=['ip.proto'], inplace=True) # Drop non-IP packets
        df.fillna(0, inplace=True) # Remaining values can be 0
        # Invalid rows from concatenation of CSV files
        errorneous_indices = df[df['ip.len'] == 'ip.len'].index
        df.drop(index=errorneous_indices, inplace=True)
    def convertColumns(self):
        print("Converting column types...")
        df = self.df  
        id_cols = self.id_cols
        # Numeric
        df[['ip.len','ip.ttl']] = df[['ip.len','ip.ttl']].astype(int)
        df['tcp.flags'] = df['tcp.flags'].astype(str).apply(int, base=16)
        df['frame.time_epoch'] = pd.to_datetime(df['frame.time_epoch'], unit='s')   
        # TCP or UDP ports
        df['srcport'] = df[['tcp.srcport','udp.srcport']].astype(int).max(axis=1)
        df['dstport'] = df[['tcp.dstport','udp.dstport']].astype(int).max(axis=1)
        df.drop(['tcp.srcport','udp.srcport', 'tcp.dstport', 'udp.dstport'], axis=1, inplace=True)
        # ID columns
        # First convert ip proto to int (this prevents duplicates)
        df['ip.proto'] = df['ip.proto'].astype(int)
        df[id_cols] = df[id_cols].astype(str)
    def partitionFlows(self):
        print("Partitioning by flow. This may take awhile...")
        df = self.df
        id_cols = self.id_cols
        fid_frame = df[id_cols].drop_duplicates() # Need unique IDs
        feature_cols = ['frame.time_epoch', 'ip.len', 'ip.ttl', 'tcp.flags'] # Raw features
        # Partition by unique ID
        partitions = []
        for i in range(fid_frame.shape[0]):
            # We must only select rows of our dataframe that match the unique ID 
            next_fid = fid_frame.iloc[i] # Next unique ID
            # Selection based on multiple column conditions being true (all ID columns must match)
            conditions = None 
            for j in range(len(id_cols)): 
                if conditions is None: # First iteration doesn't need boolean operation
                    conditions = (df[id_cols[j]] == next_fid[j])
                else:
                    conditions = conditions & (df[id_cols[j]] == next_fid[j])
            # Select raw features from matching rows
            partitions.append(df[conditions][feature_cols]) 
        self.partitions = partitions
        self.fid_frame = fid_frame
    def linkKeys(self):
        print("Linking keys to flows...")
        fid_keys = self.fid_frame.values.tolist() # Convert fid_frame to keys
        partitions = self.partitions
        # Link IDs to partitions using a dictionary
        fid_dict = {}
        indices = []
        for i in range(len(partitions)):
            if partitions[i].shape[0] >= self.threshold: # Partition contains min number of packets
                fid_dict[str(fid_keys[i])] = partitions[i]
            else:
                indices.append(i) # Otherwise we ignore this partition
        self.fid_dict = fid_dict
        # Get rid of extraneous keys (those linked to ignored partitions)
        self.keys = [v for i,v in enumerate(fid_keys) if i not in indices]
    def findIndices(self):
        print("Finding indices for subflows...")
        if self.method == "timeout":
            self.findIndicesByTimeout()
        else:  
            # Max subflow length in seconds (adjustable)
            interval = 10
            # 2D List of tuples (start and end indices) for each flow's subflows
            subflow_indices = []
            for i in range(len(self.keys)):
                # Packets in this flow
                flow_id = str(self.keys[i]) 
                flow_pkts = self.fid_dict[flow_id] 
                # Packet arrival times
                flow_pkt_times = flow_pkts['frame.time_epoch']
                start_index = flow_pkt_times.index[0]
                index_tuples = []
                while True:
                    start_time = flow_pkt_times.loc[start_index]
                    # Subtract start time from all subsequent packet times
                    sub_frame = flow_pkt_times.loc[start_index:] - start_time
                    sub_frame = sub_frame/np.timedelta64(1,'s') # Convert to seconds
                    # See if we can split
                    if (sub_frame > interval).sum() > 0:
                        end_index = sub_frame[sub_frame <= interval].index[-1] # Last index <= interval
                        index_tuples.append((start_index,end_index))
                        start_index = sub_frame[sub_frame > interval].index[0] # First index > interval
                    else:
                        # The remaining flow is <= our interval
                        end_index = sub_frame.index[-1] # Last index of sub frame
                        index_tuples.append((start_index,end_index))
                        break
                # Subflow indices for this flow
                subflow_indices.append(index_tuples) 
            self.subflow_indices = subflow_indices
    def findIndicesByTimeout(self):
        timeout_interval = 2 # Max seconds since last packet arrival (adjustable)
        subflow_indices = [] # 2D list containing subflow indices for each flow
        for i in range(len(self.keys)):  
            # Packets in this flow
            flow_id = str(self.keys[i]) 
            flow_pkts = self.fid_dict[flow_id] 
            # Packet arrival times
            flow_pkt_times = flow_pkts['frame.time_epoch']
            # Difference between any row and the row before it (arrival time difference)
            flow_time_diffs = flow_pkt_times.diff() 
            flow_time_diffs = flow_time_diffs/np.timedelta64(1,'s') # Convert to seconds
            # Indices where the inter-arrival time is greater than the timeout interval
            subflow_indices.append(flow_time_diffs[flow_time_diffs > timeout_interval].index)
        self.subflow_indices = subflow_indices
    def partitionSubflows(self):
        print("Partitioning subflows...")
        if self.method == "timeout":
            self.partitionSubflowsByTimeout()
        else:
            subflows = []
            for i in range(len(self.keys)):
                # Packets in this flow
                flow_id = str(self.keys[i])
                pkt_list = self.fid_dict[flow_id] 
                # Subflow intervals
                for j in range(len(self.subflow_indices[i])):
                    next_tuple = self.subflow_indices[i][j]
                    subflow = pkt_list.loc[next_tuple[0]:next_tuple[1]]
                    subflows.append(subflow)
            self.subflows = subflows
    def partitionSubflowsByTimeout(self):
        subflows = []
        for i in range(len(self.subflow_indices)):
            # Packets in this flow
            flow_id = str(self.keys[i])
            pkt_list = self.fid_dict[flow_id] 
            # Subflow intervals
            start = pkt_list.index[0]
            for j in range(len(self.subflow_indices[i])):
                # The end of this list is the start of the next subflow, so it must be excluded
                end = self.subflow_indices[i][j]
                subflows.append(pkt_list.loc[start:end][:-1]) # Exclude last row
                start = end
            # Final subflow for this flow
            subflows.append(pkt_list.loc[start:]) 
        self.subflows = subflows
    def extractSubflowFeatures(self):
        print("Extracting subflow features...")
        subflow_features = []
        for i in range(len(self.subflows)):
            # Next subflow
            subflow = self.subflows[i]
            num_pkts = subflow.shape[0]
            # Discard subflow if too few packets
            if num_pkts < self.threshold: 
                continue
            # Calculate duration
            start_time = subflow.iloc[0]['frame.time_epoch']
            end_time = subflow.iloc[-1]['frame.time_epoch']
            subflow_dur = end_time - start_time
            subflow_dur = subflow_dur/np.timedelta64(1,'s') # seconds
            if subflow_dur < 1:
                subflow_dur = 1
            
            # Subflow features 
            sub_features = []
            
            # Packets per second
            sub_features.append(num_pkts/subflow_dur)
            # MBits per second
            pkt_sizes = subflow['ip.len']
            total_bytes = pkt_sizes.sum()
            total_bytes /= 1e6 # Convert to MB
            bits_sec = (total_bytes * 8)/subflow_dur # MB to MBit/s
            sub_features.append(bits_sec)
            
            # Packet size statistics
            sub_features.append(pkt_sizes.mean())
            sub_features.append(pkt_sizes.std())
            sub_features.append(pkt_sizes.quantile(.25))
            sub_features.append(pkt_sizes.median())
            sub_features.append(pkt_sizes.quantile(.75))
            sub_features.append(pkt_sizes.min())
            sub_features.append(pkt_sizes.max())
            '''
            # TCP statistics
            tcp_flags = subflow['tcp.flags']
            sub_features.append(tcp_flags.mean())
            sub_features.append(tcp_flags.std())
            sub_features.append(tcp_flags.quantile(.25))
            sub_features.append(tcp_flags.median())
            sub_features.append(tcp_flags.quantile(.75))
            sub_features.append(tcp_flags.min())
            sub_features.append(tcp_flags.max())
            
            # TTL statistics
            ttl = subflow['ip.ttl']
            sub_features.append(ttl.mean())
            sub_features.append(ttl.std())
            sub_features.append(ttl.quantile(.25))
            sub_features.append(ttl.median())
            sub_features.append(ttl.quantile(.75))
            sub_features.append(ttl.min())
            sub_features.append(ttl.max())
            '''
            # No anomalies in nominal data
            sub_features.append(0)
    
            # Add sublist to main list
            subflow_features.append(sub_features)
        # Convert to dataframe
        self.subflow_features = pd.DataFrame(subflow_features)
        self.subflow_features.columns = self.feature_cols
    def shuffleSubflows(self):
        self.subflow_features = self.subflow_features.sample(frac=1)
    def featuresToCSV(self):
        print("Saving features to CSV...")
        path = self.path[0:-4]
        path += 'features/'
        file = self.file.split('.')[0]+"_features.csv"
        if not os.path.exists(path):
            os.makedirs(path)
        self.subflow_features.to_csv(path+file, encoding="utf-8", index=False)
        print(self.subflow_features.info(), end="\n\n")

# 5G Data
path = 'C:\\Users\\Michael\\Dropbox\\Backup\\Michael\\Shared\\Documents\\VTEC\\5G Code\\csv\\'
file = 'combined_5G_pcaps.csv'

method = "timeout" #Options: "timeout" or "interval" (default/recommended)

extractor = Extractor(path, file, method)

extractor.dropNaN()
extractor.convertColumns()
extractor.partitionFlows()
extractor.linkKeys()
extractor.findIndices()
extractor.partitionSubflows()
extractor.extractSubflowFeatures()
extractor.shuffleSubflows()
extractor.featuresToCSV()

# DEBUG

feats = extractor.getSubflowFeatures()
flows = extractor.getFlowInfo()