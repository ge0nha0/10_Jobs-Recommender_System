from .collect import job_simple, job_specific


def collector(args):
    df1 = job_simple(args).parse_jobs_L()
    df2 = job_specific(args, df1['구인인증번호']).parse_jobs_D()
    return 