import struct
import os


class MultiFileLogger:
    def __init__(self, log_dir):
        assert os.path.isdir(log_dir)
        if not os.path.exists(log_dir):
            os.system(f"mkdir -p {log_dir}")
        self.log_dir = log_dir
        self.files_ = {}
        self.prev_values_ = {}
        self.last_log_time_ = {}

    def log_s(self, file_name, t, x, t_label="t", x_label="x"):
        # log a 's'ingle dobule value
        if file_name not in self.files_:  # haven't logged to the given file name
            file_path = os.path.join(self.log_dir, file_name)
            f = open(file_path, "wb")
            self.files_[file_name] = f

            # Write the header
            header = f"{t_label},{x_label}\n"
            f.write(header.encode())

        # Write the data
        s = struct.pack("d" * 2, [t, x])
        self.files_[file_name].write(s)

    def log_on_change_and_timer(
        self, file_name, t, x, period, t_label="t", x_label="x"
    ):
        if (
            file_name not in self.prev_values_
            or x != self.prev_values_[file_name]
            or t - self.last_log_time_[file_name] >= period
        ):
            self.log(file_name, t, x, t_label, x_label)
            self.prev_values_[file_name] = x
            self.last_log_time_[file_name] = t

    def log_v(self, file_name, data, data_labels):
        assert len(data) == len(data_labels)
        # log a 'v'ector of double values
        if file_name not in self.files_:  # haven't logged to the given file name
            file_path = os.path.join(self.log_dir, file_name)
            f = open(file_path, "wb")
            self.files_[file_name] = f

            # Write the header
            header = ",".join([*data_labels]) + "\n"
            f.write(header.encode())

        # Write the data
        N = len(data)
        s = struct.pack("d" * N, *data)
        self.files_[file_name].write(s)

    def close_log(self, file_name):
        if file_name in self.files_:
            self.files_[file_name].close()

    def close_all_logs(self):
        for _, f in self.files_.item():
            f.close()
