require 'date'
require 'net/http'
require 'uri'
require 'json'
require 'sequel'

if (ARGV[0].nil? || ARGV[1].nil? || ARGV[2].nil? || ARGV[3].nil? || ARGV[4].nil?)
  puts "DB host [ARGV[0]] and/or name [ARGV[1]] and/or username [ARGV[2]] and/or password [ARGV[3]] and/or app_id [ARGV[4]] are nil."
  abort
end

APP_ID = ARGV[4]

base = 'USD'
currency = 'ZAR'

exchange_dates = (Date.new(1995,06,01)..Date.today).map { |date| date.to_s }

DB = Sequel.connect(adapter: 'postgresql',
                    host: ARGV[0],
                    database: ARGV[1],
                    user: ARGV[2],
                    password: ARGV[3],
                    readonly: true)

def save_data(date, base, data)
  data.each do |currency, rate|
    # DB[:exchange_rate_information].insert(base: base, currency: currency, rate: rate, created_at: date, updated_at: date)
  end
end

puts "running..."

existing_records = DB["SELECT * FROM exchange_rate_information WHERE base = 'USD' AND CURRENCY = 'ZAR'"].all

exchange_dates.each do |date|
  next if existing_records.select { |x| x if x[:date].to_s == date }.count > 0

    end_point = "https://openexchangerates.org/api/historical/#{date}.json?app_id=#{APP_ID}&base=#{base}&symbols=#{currency}"
    uri = URI(end_point)

    service_response = Net::HTTP.get_response(uri).body
    json_data = JSON.parse(service_response)['rates']

    # save_data(date, base, json_data)
end

puts "completed"
