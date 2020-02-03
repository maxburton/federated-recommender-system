def pretty_print_results(log, results, user_id):
    log.info('Recommendations for user {}:'.format(user_id))
    for row in results:
        log.info('{0}: {1}, with score {2}'.format(row[0], row[1], row[2]))
