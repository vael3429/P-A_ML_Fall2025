# GitHub Cheat Sheet
The intention of this documentation is to give you deeper guidance when it comes to using GitHub.

## Forking a repo:
Forking copies the repo you are forking from and creates a repo with the same name on your github account. This allows you to have control of a copy of the original repo without having access to write to the original repo.
* To fork a repo, you need to open the repo you want to fork in the browser.
* Find the fork button on the upper right hand side and click it.
  ![fork button](/images/fork_browser.jpg)
* The next screen is prefilled how you typically want a fork configured, so you can just push the `create fork` button.
  ![create fork](/images/create_fork.jpg)
* It will take you to the newly created fork in your github account.

## Pulling your repo down:
* click on the `< > Code` button to open the menu.
  ![Pulling down your code](/images/clone_fork.jpg)
* if you have ssh credentials configured it will have an ssh option that you cut and paste.
  * if you want to configure this, github has excellent instructions
  * first you [create an ssh key](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent)
  * then you [add it to your github account](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account)
* if you don't have ssh credentials configured for your computer, you can do it with username and password by using the HTTPS option.
* copy the text from whichever you decide to use.
* in your terminal, navigate to where you want this repo to go locally.
* run `git clone <the text you copied>`
* this will create a folder with the name of the repo with all the code in it, so you need to cd into the new directory to get to your code.

## maintaining branches:
When you first download the repo, you will be using the main branch. We want to keep the main branch, but we want to save it so that it ONLY gets updated by syncing your fork with the original. We want this because it gives you a way to download updates without re-writing or breaking your code in your branch. It won't impact other branches besides main, so you will have control over which branches you put the updated code in.

To see what branches you have locally, run `git branch`.

### syncing fork
* go to your fork in your github account in the browser.
* click the button with a circular recycle symbol that says `Sync fork`
* That will open a dropdown menu:
  * If your main branch is up to date with the original repo's main branch so you don't need to sync. There will be no buttons and it will have the text:
    > No new commits to fetch. Enjoy your day!
  * If there are changes, there are two options:
    * preview the changes - this just shows you what has changed since you last synced your main branches
    * sync the repo - this goes ahead and updates your code in github to include the changes
* after you sync the fork in the browser, you have to pull the code down locally.
* in your terminal, in the project directory, while on the main branch, run `git pull origin main`
* you now have the updated code in your main branch.

### creating a branch:
It's a good thing to check what the status of your files is regularly, especially when changing branches, committing code, and updating code. To check if you have any pending changes or staged files (that means changes that you have added in but haven't committed yet), run `git status`. Any red things are not checked in. That means it can either move with you accidentally if you change to a different branch or it may prevent you from being able to switch branches. We'll go over what to do if that happens in the next section.

If you are okay to create a new branch, you type `git checkout -b the_name_of_your_new_branch`. It will create the branch and you will now be making changes in that branch instead of the main branch.

### dealing with uncommitted changes:
If you are trying to switch branches or you run git status and see that you have unchecked files, you have a few options:
1. You can preview your changes and stage them for commit
  * run `git add -p` and that will give you a diff for each chunk of changes and you can say if you want to add it or if you don't.
  * run `git status` to make sure all the changes you want to add are green.
  * if you created new files, they aren't automatically tracked, so you may have to run `git add <filename>` to stage them.
  * once all the changes you want to commit are staged, you can run `git commit -m "something describing your change"`
  * USE WITH CAUTION! if there was code you didn't stage and just want to get rid of, you can run `git checkout .` to delete anything that isn't staged or committed. If you run this without staging files first, it will delete all changes you have made and you can't get them back. So only run this once you have staged and committed any changes you DO want to make.
2. You can stash your changes. The stash is a place where you can stuff code changes that you aren't ready to stage or commit but you want to still have ready to add back in whenever you want. For example, if you are on the branch you created but you want to go back to the main branch and pull down updates after syncing, if you have unstaged work, it won't let you do this because it doesn't know how to handle your unstaged changes.

You can see what is in your stash by running `git stash list`. There can be multiple things in your stash, so each slot will have an identifier like `stash@{0}`. If you want to add a particular stash, you can either pop it or apply it. If you pop it, it removes it from your stash. You pop by running `git stash pop stash@{0}` using the identifier of the stash you want to use. If you want to apply it, the changes will be applied to your branch but will also stay in your stash. You can do this by running `git stash apply stash@{0}`.

### switching branches:
Before you switch branches, it's a good idea to run `git status` to make sure you aren't going to inadvertently bring changes with you. Deal with any unstaged changes before switching. Changed files will prevent you from changing branches, but newly added files will just sneak to the branch with you. To switch to another branch (that already exists) run `git checkout <branch name>`. It's very similar to the command to create a branch, the difference is the `-b` flag. That is what tells it to create a branch vs switch to a branch.

### rebasing:
This can be an intimidating thing initially. It's a complicated concept to understand. To get changes from the branch yours is based off into yours, you need to pull them in. There are two ways of doing this, merging and rebasing. Merging just jams all the changes you don't have from that branch into yours, whereas rebasing rewrites the history of the branch in the order the changes were made. Rebasing is preferred because it preserves the history better.

If you update your main branch by syncing in the browser and pulling down those changes, you still only have those updates in the main branch. If you want to also include them in your other branch(es), you need to manually do that by rebasing. You would make sure you are in the branch you want to update and then you would run `git rebase main`. Most of the time this works without a problem, but sometimes github gets changes from the update in the same place you've made changes in your branch and it can't determine what to do. This is called a merge conflict. If this happens, don't panic! It gives instructions about what to do in the error message. We're going to go through how to read this message:

```
➜  P-A_ML_Fall2025 git:(testing/rebase) git rebase main
Auto-merging README.md
CONFLICT (content): Merge conflict in README.md
error: could not apply f4eca643... alter readme
hint: Resolve all conflicts manually, mark them as resolved with
hint: "git add/rm <conflicted_files>", then run "git rebase --continue".
hint: You can instead skip this commit: run "git rebase --skip".
hint: To abort and get back to the state before "git rebase", run "git rebase --abort".
Could not apply f4eca643... alter readme
```

1. The first line is where I ran the rebase.
2. the step it was on was trying to Auto-merge the README.md file
3. what happened is a conflict, it then describes which file the conflict is in.
4. It's telling you that this made it error out and it tells you the commit message of the commit that it choked on.
5. The first hint it gives is that it's not going to be doing this work. You are going to have to resolve the conflict. It doesn't give good details on what that entails though, so we're going to take it step by step before moving to the next hint. When a conflict happens, it will insert code marking the conflict. What you are looking for is added code that looks like this:

```
<<<<<<< HEAD
The change you have
=======
The change that was made in main
>>>>>>> main
```

This means it is up to you to decide what stays and what goes. You look at the code in both changes and determine what you want to stay. You remove the conflict marker and leave what you decide needs to be left. So for example, if we fixed the conflict above and decided both lines should be in your branch, you would just have:

```
The change you have
The change that was ade in main
```

Make sure you save the file after making your changes. Now we're going to go back to the hints to figure out what to do next.

6. it tells us that all the files we fixed need to be added back in by either using `git add <filename>` to add them or `git rm <filename>` to remove it. It also tells us that once we've added the file(s) back in, we need to tell it to continue the rebase by running `git rebase --continue`.
7. If you decide it's just not a chunk of code you'd like added to your branch, you can just skip the problematic commit. This can cause conflicts in future commits being applied in this rebase, so it can be annoying if you have to keep skipping the same conflict.
8. You don't have to continue with the rebase at all if you don't want to! If you think you messed something up, no harm no foul, just run `git rebase --abort` to make it like you never rebased at all.
9. Then it lets me know once again that it couldn't apply the commit with that sha (the numbers and letters to identify each commit) and tells me the commit message too.

You can rerun the rebase if it failed and you want to try again. It also may have conflicts in multiple commits, so you may encounter this screen again with other commits included in the rebase. You just look at what files are affected, fix the conflict, add the change back in, and continue the rebase each time.

### commits:
Think of commits as save points. If you get your code working but aren't totally finished with it, but you'd like to be able to get back to this point again if you need to, make a commit. When you make the commit, use the `-m` flag to tell it you want to give it a message, then in quotation marks you write a short description about what change you made. Remember how when there is a merge conflict in the rebase it tells us which commit failed? Make your message something that is meaningful for that chunk of work. You can also remove commits if you want to. So if you make sure that the work that is contained in a commit is fairly small and is a good save point, you will always be able to revert back to how the code was at any of those commits. This means you can make a commit, then make code changes that could be disasterous and if so, you just run `git checkout .` to delete all the changes you've made and you are back to the last commit. If you worked for days and days on all kinds of different changes and then you wanted to delete just some of them, you would have to go in and manually make it how you want it instead of just reverting back to a previous version and that can be much more complicated and risky. And remember, if you run `git checkout .` it removes any changes you have made, it blows it away and you can't get it back.

### interactive rebasing:
This is a little advanced, so it can be scary till you get used to the concept. Don't worry though because you can always abort it at any time. An interactive rebase is similar to a normal rebase, but it lets you decide what to do with each commit individually. The command is `git rebase -i HEAD~N`. This means we're telling git we're going to re-write history using a rebase and the `-i` flag tells it we want to do it interactively. The last part needs to be adjusted according to your needs. `HEAD~N` is telling it how many commits to go back. So let's say you've committed 10 times and you want to manage 10 past commits to this branch, you would use `git rebase -i HEAD~10`. When you run that, it will take the history of the past 10 commits and give you options for what you can do with them.

| option (shorthand)| meaning |
| --- | --- |
| pick (p) | You want to keep this commit without any changes. |
| reword (r) | You want to keep the commit but change the message for that commit. |
| edit (e)| You want to apply this commit but you want to stop the rebase after this point so you can make changes (like fixing a typo or an error) before continuing the rebase |
| squash (s) | You can combine this commit in with the previous one which turns them into one commit. |
| fixup (f) | It works like squash does but it removes the message for this commit in favor of the previous commit's message. |
| drop (d) | Remove this commit like it never happened. |

There will be steps in this process that come up depending on what you want to do. You will have the option to edit commit messages, but you don't have to make any changes. If a vim style session starts and expects you to enter something, you can just press `:q` to quit out of the session and it'll keep what it showed you.

## pushing code:
Pushing code to your branch is easy. After you make a commit, you run `git push origin <branch name>` to push it up. Those changes will now be in github. If you navigate to your fork of the repo, you will see a yellow banner that lets you know something has changed and asking you if you want to make a pull request.

## pull requests:
A pull request allows other people to see what changes you are proposing and ask questions, request changes, or just discuss what is happening on any particular line. Someone who is reviewing your code can approve it (so it can be merged in) or reject it (which blocks the ability to commit till that person re-reviews the code after changes are made).

You can make a pull draft pull request if you want people to know that it's not ready for review yet.

You can create a pull request at any time to facilitate collaboration. If you make some changes and you're not sure they're doing what you think they're doing and you want someone to be able to look at your changes and give you advice, you make a pull request and send them a link to it. It's a very handy way for them to see what you've done without having to dig through the codebase trying to find it.
