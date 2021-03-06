from ..settings import list_settings, read_value, write_value, BADGER_PATH_DICT
import os
import logging
logger = logging.getLogger(__name__)
from ..utils import yprint


def config_settings(args):
    key = args.key

    if key is None:
        yprint(list_settings())
        return

    try:
        print('')
        return _config_path_var(key)
    except IndexError:
        pass
    except KeyboardInterrupt:
        return

    logger.error(f'{key} is not a valid Badger config key!')


def _config_path_var(var_name):
    display_name = BADGER_PATH_DICT[var_name]['display name']
    desc = BADGER_PATH_DICT[var_name]['description']

    print(f'=== Configure {display_name} ===')
    print(f'*** {desc} ***\n')
    while True:
        res = input(
            f'Please type in the path to the Badger {display_name} folder (S to skip, R to reset): \n')
        if res == 'S':
            break
        if res == 'R':
            _res = input(
                f'The current value {read_value(var_name)} will be reset, proceed (y/[n])? ')
            if _res == 'y':
                break
            elif (not _res) or (_res == 'n'):
                print('')
                continue
            else:
                print(f'Invalid choice: {_res}')

        res = os.path.abspath(os.path.expanduser(res))
        if os.path.isdir(res):
            _res = input(f'Your choice is {res}, proceed ([y]/n)? ')
            if _res == 'n':
                print('')
                continue
            elif (not _res) or (_res == 'y'):
                break
            else:
                print(f'Invalid choice: {_res}')
        else:
            _res = input(
                f'{res} does not exist, do you want to create it ([y]/n)? ')
            if _res == 'n':
                print('')
                continue
            elif (not _res) or (_res == 'y'):
                os.makedirs(res)
                print(f'Directory {res} has been created')
                break
            else:
                print(f'Invalid choice: {_res}')

    if res == 'R':
        write_value(var_name, None)
        print(f'You reset the Badger {display_name} folder setting')
    elif res != 'S':
        write_value(var_name, res)
        print(f'You set the Badger {display_name} folder to {res}')
