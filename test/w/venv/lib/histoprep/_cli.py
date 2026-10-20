import os, sys
class _Cmd:
    def main(self, args, prog_name):
        with open(os.environ["LOGF"], "a") as f:
            f.write("HISTOPREP " + " ".join(args) + "\n")
        assert sum(1 for a in args if not a.startswith("--")) == len(args) - sum(1 for a in args if a.startswith("--")) 
        out = args[args.index("--output") + 1]
        assert "*" in args[args.index("--input") + 1]
        for s in ("a", "b"):
            os.makedirs(os.path.join(out, s, "tiles"), exist_ok=True)
            open(os.path.join(out, s, "metadata.parquet"), "w").close()
            open(os.path.join(out, s, "tiles", "x0_y0_w512_h512.jpeg"), "w").close()
        sys.stdout.write("Cutting  10%\rCutting  50%\rCutting 100%\r\nok\r\n")
cut_slides = _Cmd()
