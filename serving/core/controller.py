import re
from .logger import get_logger


class AstraSimChildDied(RuntimeError):
    """The ASTRA-Sim child exited while the frontend was waiting for it.

    deviations.md D25-b. `p.stdout.readline()` returns "" forever once the child is
    gone, and "" matches neither of the waiting loops' exit conditions, so the
    frontend used to spin at 100% CPU growing its output list without bound
    (measured: RSS 6.51 -> 6.68 GB in 30 s, child a zombie at exit 1) until the
    caller's timeout fired -- hours later, and reported as a slow simulation.

    Raising instead turns that into an immediate, explained failure, which the
    planner already classifies correctly: `planner/predictor/llmservingsim.py`
    records a non-zero exit with its stderr tail as a failure outcome, distinct
    from TIMEOUT.
    """


class Controller():
    def __init__(self, total_num):
        self.end_dict = {}
        self.total_num = total_num
        self.logger = get_logger(self.__class__)
        for i in range(total_num):
            self.end_dict[i] = -1


    #: Consecutive empty reads tolerated from a child that is still running. A real
    #: blank line comes back as "\n", so "" means the stream is at EOF; this only
    #: covers the narrow case of a closed stream on a process that has not reaped
    #: yet, where poll() is still None.
    _MAX_EMPTY_READS = 64

    def _check_child_alive(self, p, empty_reads, where):
        """Raise if the child is gone, or if its stream died while it lingers.

        D25-b. No cap is placed on the output list itself -- this exception is the
        bound. A length cap would need an assumption about how much a healthy run
        prints, and healthy runs print a lot.
        """
        rc = p.poll()
        if rc is not None:
            raise AstraSimChildDied(
                f"ASTRA-Sim exited with code {rc} while the frontend was waiting in "
                f"{where}(). {self._stderr_hint(p)}"
            )
        if empty_reads >= self._MAX_EMPTY_READS:
            raise AstraSimChildDied(
                f"ASTRA-Sim gave {empty_reads} consecutive empty reads in {where}() "
                f"while still running; its stdout is unusable. {self._stderr_hint(p)}"
            )

    @staticmethod
    def _stderr_hint(p):
        """Quote the child's stderr, which D25-b redirects to a file in the run tree.

        Before D25-b stderr was a pipe nobody ever read, so the one line that
        explained the failure died with it.
        """
        path = getattr(p, "astra_stderr_path", None)
        if not path:
            return "The child's stderr was not captured to a file."
        try:
            with open(path, encoding="utf-8", errors="replace") as f:
                tail = f.read()[-4000:].strip().splitlines()[-20:]
        except OSError as exc:
            return f"Could not read the child's stderr at {path}: {exc}"
        if not tail:
            return f"The child's stderr at {path} is empty."
        return "Last lines of {}:\n{}".format(path, "\n".join(tail))

    def read_wait(self, p):
        out = [""]
        empty_reads = 0
        while "Waiting" not in out[-1] and out[-1] != "Checking Non-Exited Systems ...\n":
            line = p.stdout.readline()
            if line == "":
                empty_reads += 1
                self._check_child_alive(p, empty_reads, "read_wait")
            else:
                empty_reads = 0
            # For debugging
            # print(line, end='')
            out.append(line)
            p.stdout.flush()
        return out

    def check_end(self, p):
        out = ["",""]
        empty_reads = 0
        while out[-2] != "All Request Has Been Exited\n" and out[-2] != "ERROR: Some Requests Remain\n":
            line = p.stdout.readline()
            if line == "":
                empty_reads += 1
                self._check_child_alive(p, empty_reads, "check_end")
            else:
                empty_reads = 0
            out.append(line)
            p.stdout.flush()
        print(out[-4], end='')
        print(out[-2], end='')
        return out

    def write_flush(self, p, input):
        # For debugging
        # print(input)
        # A dead child is as likely to be noticed here as in the read loops --
        # whichever comes first after it goes. Python's own BrokenPipeError already
        # ends the run promptly, but it names neither the child's exit code nor its
        # stderr, which is the half of D25-b that makes the failure readable. So
        # translate it. Measured: killing AnalyticalAstra mid-run surfaced here, not
        # in read_wait.
        try:
            p.stdin.write(input+'\n')
            p.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            rc = p.poll()
            raise AstraSimChildDied(
                f"ASTRA-Sim's stdin is closed ({type(exc).__name__}: {exc}); the child "
                f"exited with code {rc} while the frontend was writing to it. "
                f"{self._stderr_hint(p)}"
            ) from exc
        return

    def parse_output(self, output):
        pattern = r"sys\[(\d+)\] iteration (\d+) finished, (\d+) cycles, exposed communication (\d+) cycles."
        match = re.search(pattern, output)
        if match:
            sys = int(match.group(1))
            id = int(match.group(2))
            cycle = int(match.group(3))
            com_cycle = int(match.group(4))

            if self.end_dict[sys] != id:
                self.logger.info(
                    "NPU[%d] iteration %d finished, %d cycles, exposed communication %d cycles.",
                    sys,
                    id,
                    cycle,
                    com_cycle,
                )
                self.end_dict[sys] = id
            return {'sys': sys, 'id': id, 'cycle': cycle}
        return