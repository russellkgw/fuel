require 'date'
require 'net/http'
require 'uri'
require 'json'
require 'sequel'

if ARGV[0].nil?
  puts "Error: Exchange rate app id not supplied."
  abort
end

APP_ID = ARGV[0]

base = 'USD'
currency = 'ZAR'

# exchange_dates = (Date.new(1995,12,01)..Date.today).map { |date| date.to_s }
exchange_dates = (Date.new(1999,01,01)..Date.new(1999,01,31)).map { |date| date.to_s }

DB = Sequel.connect(adapter: 'postgresql',
                    host: 'localhost',
                    database: 'fuel_development',
                    user: 'postgres',
                    password: nil,
                    readonly: true)

def save_data(date, base, data)
  data.each do |currency, rate|
    DB[:exchange_rates].insert(base: base, currency: currency, value: rate, date: date, source: 'https://openexchangerates.org/')
  end
end

puts "running..."

existing_records = DB["SELECT * FROM exchange_rates WHERE base = 'USD' AND CURRENCY = 'ZAR'"].all

puts existing_records.count.to_s

exchange_dates.each do |date|
  next if existing_records.select { |x| x if x[:date].to_s == date }.any?

    # end_point = "https://openexchangerates.org/api/historical/#{date}.json?app_id=#{APP_ID}&base=#{base}&symbols=#{currency}"
    # uri = URI(end_point)
    #
    # service_response = Net::HTTP.get_response(uri).body
    # json_data = JSON.parse(service_response)['rates']

    # save_data(date, base, json_data)
end

puts "completed"
