# Kept just for record.
# Do not run.

require 'fileutils'

def colorize(text, color_code)
  "\e[#{color_code}m#{text}\e[0m"
end

def red(text); colorize(text, 31); end
def green(text); colorize(text, 32); end

puts red("Error: This script is used only for correction happened on 12/7/2016.")
exit

def correct_under_folder(folder,name)
    paths = Dir.glob(File.join(folder,"/**/*.csv"))
    perm = {'0.1' => '0.4',"-0.1" => "0.2", "-0.3" => "0", "-0.5" => "-0.2", "-0.7" => "-0.5"}

    count = 0

    paths.each{|path|
        if path =~ /red.+\.csv$/
            path =~ /red (-?.+)\.csv$/
            raise "No voltage value: #{path}" if $1.nil?
            if perm[$1]
                new_path = path.gsub(/red (-?.+)\.csv$/) {
                    "red #{perm[$1]}.csv"
                    }.gsub(name,name+"_new")
            else
                raise "No voltage value: #{path}" if $1.nil?
            end
            FileUtils.mkdir_p(File.dirname(new_path))
            FileUtils.cp(path,new_path)
            count += 1
        else
            new_path = path.gsub(name,name+"_new")
            FileUtils.mkdir_p(File.dirname(new_path))
            FileUtils.cp(path,new_path)
            count += 1
        end
    }

    puts green("#{count} files were copied.")
end

correct_under_folder("../data/kinetics/split","split")
correct_under_folder("../data/kinetics/fitted_manual","fitted_manual")

