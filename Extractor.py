import pandas as pd
import numpy as np
import os

class Extractor:
    def __init__(self, path, method, style):
        self.path = path
        self.method = method
        self.threshold = 2 # Min packets for flow analysis 
        # Options: ip.src, ip.dst, ip.proto, srcport, dstport
        self.id_cols = ['ip.dst', 'ip.proto']
        # Can choose KB or MB
        self.bits_col = "KBits_Per_Sec" if style=="kb" else "MBits_Per_Sec"
        self.bytes_convert = 1e3 if style=="kb" else 1e6
        self.feature_cols = ['Pkts_Per_Sec', self.bits_col, 'Pkt_Size_Avg', 'Pkt_Size_Std', 'Pkt_Size_Q1', 'Pkt_Size_Q2', 'Pkt_Size_Q3', 
                        'Pkt_Size_Min', 'Pkt_Size_Max', 'ACK_Sec', 'SYN_Sec', 'TTL_Avg']
    def findCSVFiles(self):
        files = []
        for file in os.listdir(path):
            ext = file[-3:]
            if ext == "csv":
                files.append(file)
        self.files = files
    def getCSVFiles(self):
        return self.files
    def loadCSV(self, file):
        print(f"Loading: {file}")
        self.df = pd.read_csv(self.path+file)
        self.file = file
    def getDF(self):
        return self.df
    def dropNaN(self):
        print("Cleaning data...")
        df = self.df
        # NAN values
        df.dropna(subset=['ip.proto'], inplace=True) #Non-IP packets
        df.fillna(0, inplace=True) #Remaining values can be 0
        # Invalid rows are from concatenation of CSV files
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
        df[id_cols] = df[id_cols].astype(str)
    def partitionFlows(self):
        print("Partitioning by flow. This may take awhile...")
        df = self.df
        fid_frame = df[self.id_cols].drop_duplicates() # Need unique IDs
        id_cols = self.id_cols
        feature_cols = ['frame.time_epoch', 'ip.len', 'ip.ttl', 'tcp.flags'] # Raw features
        # Partition by unique ID
        partitions = []
        for i in range(fid_frame.shape[0]):
            # We must only select rows of our dataframe that match the unique ID 
            next_fid = fid_frame.iloc[i] # Next unique ID
            conditions = None # Selection based on multiple column conditions being true (all ID columns must match)
            for j in range(len(id_cols)): 
                if conditions is None: # First iteration doesn't need boolean operation
                    conditions = (df[id_cols[j]] == next_fid[j])
                else:
                    conditions = conditions & (df[id_cols[j]] == next_fid[j])
            partitions.append(df[conditions][feature_cols]) # Select raw features from matching rows
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
        self.new_keys = [v for i,v in enumerate(fid_keys) if i not in indices]
    def findIndices(self):
        print("Finding indices for subflows...")
        if self.method == "timeout":
            self.findIndicesByTimeout()
        else:  
            # Partition flows into fixed time intervals
            new_keys = self.new_keys
            interval = 10 # Max subflow length in seconds (adjustable)
            # 2D lists contain starting and ending subflow indices for each flow
            first_indices  = [] 
            last_indices = [] 
            for i in range(len(new_keys)):
                flow_id = str(new_keys[i]) # Next ID
                flow_pkts = self.fid_dict[flow_id] # Packets for this flow
                flow_packet_times = flow_pkts['frame.time_epoch'] # Packet arrival times
                index = flow_packet_times.index[0] # Starting index
                f_indices = [index]
                l_indices = []
                while True:
                    start_time = flow_packet_times.loc[index] # Start time
                    # Subtract start time from all subsequent packet times
                    sub_frame = flow_packet_times.loc[index:] - start_time
                    sub_frame = sub_frame/np.timedelta64(1,'s') # Convert to seconds
                    # See if we can split
                    if (sub_frame > interval).sum() > 0:
                        index = sub_frame[sub_frame <= interval].index[-1] # Last index <= interval
                        l_indices.append(index)
                        index = sub_frame[sub_frame > interval].index[0] # First index > interval
                        f_indices.append(index)
                    else:
                        # The remaining flow is <= our interval
                        l_indices.append(sub_frame.index[-1]) # Last index of sub frame
                        break
                # Lists of first and last subflow indices for this flow
                first_indices.append(f_indices)
                last_indices.append(l_indices)
            self.subflow_indices = [first_indices,last_indices] # List of 2D lists!
    def findIndicesByTimeout(self):
        new_keys = self.new_keys
        timeout_interval = 2 # Max seconds since last packet arrival (adjustable)
        subflow_indices = [] # 2D list containing subflow indices for each flow
        for i in range(len(new_keys)):  
            flow_id = str(new_keys[i])
            flow_pkt_times = self.fid_dict[flow_id]['frame.time_epoch'] # Packet arrival times for this flow
            flow_time_diffs = flow_pkt_times.diff() # Difference between any row and the row before it (arrival time difference)
            flow_time_diffs = flow_time_diffs/np.timedelta64(1,'s') # Convert to seconds
            # Indices where the inter-arrival time is greater than the timeout interval
            subflow_indices.append(flow_time_diffs[flow_time_diffs > timeout_interval].index)
        self.subflow_indices = subflow_indices
    def partitionSubflows(self):
        print("Partitioning subflows...")
        if self.method == "timeout":
            self.partitionSubflowsByTimeout()
        else:
            new_keys = self.new_keys
            # Starting and ending subflow indices for each flow
            first_indices = self.subflow_indices[0]
            last_indices = self.subflow_indices[1]
            # Partition flows into subflows using indices
            subflows = [] 
            for i in range(len(new_keys)):
                next_key = str(new_keys[i])
                pkt_list = self.fid_dict[next_key] # Packets in this flow
                # Subflow intervals
                for j in range(len(last_indices[i])):
                    next_start = first_indices[i][j]
                    next_end = last_indices[i][j]
                    subflow = pkt_list.loc[next_start:next_end]
                    subflows.append(subflow)
            self.subflows = subflows
    def partitionSubflowsByTimeout(self):
        subflow_indices = self.subflow_indices
        new_keys = self.new_keys
        # Partition flows using subflow indices
        subflows = []
        for i in range(len(subflow_indices)):
            next_key = str(new_keys[i])
            pkt_list = self.fid_dict[next_key] # Packets in this flow
            # Subflow intervals
            next_start = pkt_list.index[0]
            for j in range(len(subflow_indices[i])):
                next_end = subflow_indices[i][j] # Start of next subflow
                subflows.append(pkt_list.loc[next_start:next_end][:-1]) # Exclude next_end
                next_start = next_end
            subflows.append(pkt_list.loc[next_start:])
        self.subflows = subflows
    def extractSubflowFeatures(self):
        print("Extracting subflow features...")
        subflow_features = []
        for i in range(len(self.subflows)):
            subflow = self.subflows[i]
            sub_features = []
            num_pkts = subflow.shape[0]
            if num_pkts < self.threshold: # Discard subflow if too few packets
                continue
            # Packet sizes and duration
            pkt_sizes = subflow['ip.len']
            start_time = subflow.iloc[0]['frame.time_epoch']
            end_time = subflow.iloc[-1]['frame.time_epoch']
            subflow_dur = end_time - start_time
            subflow_dur = subflow_dur/np.timedelta64(1,'s') # seconds
            if subflow_dur < 1:
                subflow_dur = 1
            # Packets per second
            sub_features.append(num_pkts/subflow_dur)
            # Data rate (bits per second)
            total_bytes = pkt_sizes.sum()
            total_bytes /= self.bytes_convert # K or M
            bits_sec = (total_bytes * 8)/subflow_dur
            sub_features.append(bits_sec)
            # Avg size of packets in subflow (bytes)
            avg_bytes = pkt_sizes.mean()
            sub_features.append(avg_bytes)
            # Standard deviation of frame size
            sub_features.append(pkt_sizes.std())
            # Inter-quartile range
            sub_features.append(pkt_sizes.quantile(.25))
            sub_features.append(pkt_sizes.median())
            sub_features.append(pkt_sizes.quantile(.75))
            # Min and max  sizes
            sub_features.append(pkt_sizes.min())
            sub_features.append(pkt_sizes.max())
            # TCP Flags
            ACK_mask = 16
            SYN_mask = 2
            num_acks = ((subflow['tcp.flags'] & ACK_mask) > 0).sum()
            num_syns = ((subflow['tcp.flags'] & SYN_mask) > 0).sum()
            sub_features.append(num_acks/subflow_dur)
            sub_features.append(num_syns/subflow_dur)
            # IP Avg TTL
            sub_features.append(subflow['ip.ttl'].mean())
            # Add sublist to main list
            subflow_features.append(sub_features)
        self.df_subflows = pd.DataFrame(subflow_features)
        self.df_subflows.columns = self.feature_cols
    def generate_mal_subflows(self):
        print("Generating malicious subflows...")
        df_subflows = self.df_subflows
        df_subflows['Malicious'] = 0
        # Number of malicious subflows to generate
        num_mal = np.ceil(df_subflows.shape[0] / 5).astype(int)
        malicious_subflows = []
        for i in range(num_mal):
            malicious_subflows.append(self.mal_subflow())
        malicious_subflows = pd.DataFrame(malicious_subflows)
        malicious_subflows.columns = self.feature_cols
        malicious_subflows['Malicious'] = 1
        # Concatenate normal subflows and malicious subflows
        data = [df_subflows, malicious_subflows]
        self.df_subflows = pd.concat(data)
    def mal_subflow(self):
        mal_features = []
        # Packets per second should be high, at least 6000
        num_pkts = np.random.randint(30000,35000)
        dur = 5
        pkts_sec = num_pkts/dur
        mal_features.append(pkts_sec)
        # Simple ping flood with same size packets (bytes)
        pkt_size = 64 # Can be as high as 65,535 in IPv4
        total_size = pkt_size * num_pkts
        total_size /= self.bytes_convert
        bits_sec = (total_size * 8)/dur
        mal_features.append(bits_sec)
        mal_features.append(pkt_size) 
        mal_features.append(0) # no standard deviation for uniform sizes
        for i in range(5):
            mal_features.append(pkt_size)
        mal_features.append(0) # No SYN or ACKs in ICMP ping
        mal_features.append(0)
        mal_features.append(64) # Arbitrary TTL
        return pd.Series(mal_features)
    def shuffleSubflows(self):
        self.df_subflows = self.df_subflows.sample(frac=1)
    def featuresToCSV(self):
        print("Saving features to CSV...")
        path = self.path[0:-4]
        path += 'features/'
        file = self.file.split('.')[0]+"_features.csv"
        if not os.path.exists(path):
            os.makedirs(path)
        self.df_subflows.to_csv(path+file, encoding="utf-8", index=False)
        print(self.df_subflows.info(), end="\n\n")

# Clean 5G (Ashley)
path = '/smallwork/m.hackett_local/data/ashley_pcaps/captures/csv/' 

# MODBUS Testing
#path = '/smallwork/m.hackett_local/data/modbus/clean/csv/' # Clean modbus
#path = '/smallwork/m.hackett_local/data/modbus/pingFloodDDoS/csv/' # Attack modbus (ping flood)
#path = '/smallwork/m.hackett_local/data/modbus/modbusQueryFlooding/csv/' # Attack modbus (query flood)
#path = '/smallwork/m.hackett_local/data/modbus/tcpSYNFloodDDoS/csv/' # Attack modbus (query flood)

method = "timeout" #Options: "timeout" or "interval" (default)
style = "mb" #Options: "kb" or "mb" (default)
gen_malicious = True #Generate malicious subflows (5G data only)

extractor = Extractor(path, method, style)
extractor.findCSVFiles()
files = extractor.getCSVFiles()

for file in files:
    extractor.loadCSV(file)
    extractor.dropNaN()
    extractor.convertColumns()
    extractor.partitionFlows()
    extractor.linkKeys()
    extractor.findIndices()
    extractor.partitionSubflows()
    extractor.extractSubflowFeatures()
    if gen_malicious:
        extractor.generate_mal_subflows()
    extractor.shuffleSubflows()
    extractor.featuresToCSV()