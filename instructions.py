def task1(thread):

  # do some task with thread here
  # e.g.
  res = thread.ramses_data["hydro"]["density"].max().to("g/cm^3")
  return res